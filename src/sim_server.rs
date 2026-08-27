//! Headless MuJoCo simulation server with a browser-based viewer.
//!
//! HTTP viewer:
//!     http://<host>:9000
//!
//! WebSocket:
//!     ws://<host>:9001
//!
//! The simulation and renderer run headlessly. Rendered RGB frames are
//! JPEG-encoded and streamed to connected WebSocket clients.
//!
//! Usage:
//!
//! ```text
//! MUJOCO_STATIC_LINK_DIR=/mujoco/build/lib \
//!     cargo run --example sim_server --features networking -- /path/to/model.xml
//! ```
//!
//! With no model argument, a small built-in demo model is used.

use mujoco_rs::renderer::MjRenderer;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use image::codecs::jpeg::JpegEncoder;
use mujoco_rs::prelude::*;
use tungstenite::{Error as WsError, Message, WebSocket, accept};
const HOST: &str = "0.0.0.0";

const HTTP_PORT: u16 = 9000;
const WS_PORT: u16 = 9001;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

const RENDER_FPS: f64 = 30.0;
const JPEG_QUALITY: u8 = 80;

const MAX_SIM_STEPS_PER_FRAME: u32 = 64;

// Server -> client message tags.
//
// FRAME:
//   1 byte  tag
//   4 bytes little-endian payload length
//   2 bytes little-endian width
//   2 bytes little-endian height
//   JPEG bytes
//
// STATE:
//   1 byte  tag
//   4 bytes little-endian payload length
//   JSON bytes

const S2C_FRAME: u8 = 0x80;
const S2C_STATE: u8 = 0x81;

const VIEWER_HTML: &str = include_str!("viewer/index.html");

const DEMO_MODEL: &str = r#"
<mujoco>
    <visual>
        <global offwidth="1280" offheight="720"/>
        <headlight
            diffuse="0.8 0.8 0.8"
            ambient="0.3 0.3 0.3"
            specular="0.1 0.1 0.1"/>
    </visual>

    <worldbody>
        <light
            pos="0 0 3"
            dir="0 0 -1"
            diffuse="1 1 1"
            ambient="0.3 0.3 0.3"
            castshadow="true"/>

        <geom
            name="floor"
            type="plane"
            size="10 10 1"
            rgba="0.8 0.8 0.8 1"/>

        <body name="ball" pos="0 0 1.5">
            <geom
                name="ball_geom"
                type="sphere"
                size=".15"
                rgba="0 0.8 0 1"/>

            <joint
                name="ball_joint"
                type="free"/>
        </body>
    </worldbody>
</mujoco>
"#;

#[derive(Clone)]
struct Frame {
    seq: u64,
    width: u16,
    height: u16,
    jpeg: Vec<u8>,
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, image::ImageError> {
    let mut buf = Vec::with_capacity((width as usize) * (height as usize) / 10);

    let mut encoder = JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);

    encoder.encode(rgb, width, height, image::ColorType::Rgb8.into())?;

    Ok(buf)
}

fn send_frame(ws: &mut WebSocket<TcpStream>, frame: &Frame) -> Result<(), WsError> {
    let payload_len = 4 + frame.jpeg.len();

    let mut buf = Vec::with_capacity(5 + payload_len);

    buf.push(S2C_FRAME);

    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());

    buf.extend_from_slice(&frame.width.to_le_bytes());

    buf.extend_from_slice(&frame.height.to_le_bytes());

    buf.extend_from_slice(&frame.jpeg);

    ws.send(Message::Binary(buf.into()))
}

fn send_state(ws: &mut WebSocket<TcpStream>, time: f64) -> Result<(), WsError> {
    let payload = format!(r#"{{"time":{time}}}"#);

    let mut buf = Vec::with_capacity(5 + payload.len());

    buf.push(S2C_STATE);

    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());

    buf.extend_from_slice(payload.as_bytes());

    ws.send(Message::Binary(buf.into()))
}

/// Handle one WebSocket client.
///
/// The WebSocket listener is on a dedicated port, so there is no need for
/// any HTTP routing or request parsing here. Tungstenite receives the raw
/// TCP stream and performs the WebSocket handshake itself.
fn serve_client(
    stream: TcpStream,
    latest: &Mutex<Option<Frame>>,
    last_time: &Mutex<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tungstenite's HTTP/WebSocket handshake is blocking, so the
    // socket must remain blocking while accept() performs it.
    let mut ws = accept(stream)?;

    println!("WebSocket client connected");

    // After the handshake has completed, we can switch the underlying
    // TCP socket to nonblocking mode for polling client messages.
    ws.get_mut().set_nonblocking(true)?;

    let mut last_seq = 0u64;
    let mut last_sent_time = f64::NAN;

    loop {
        //
        // Check for incoming client messages.
        //
        // The socket is nonblocking, so read() returns WouldBlock when
        // there is currently no client message.
        //
        match ws.read() {
            Ok(Message::Close(_)) => {
                println!("WebSocket client disconnected");
                return Ok(());
            }

            Ok(Message::Ping(payload)) => {
                ws.send(Message::Pong(payload))?;
            }

            Ok(Message::Pong(_)) => {}

            Ok(Message::Text(_)) => {}

            Ok(Message::Binary(_)) => {}

            Ok(Message::Frame(_)) => {}

            Err(WsError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No incoming data.
            }

            Err(WsError::ConnectionClosed) => {
                println!("WebSocket client disconnected");
                return Ok(());
            }

            Err(e) => {
                return Err(Box::new(e));
            }
        }

        //
        // Send the latest rendered frame.
        //
        {
            let guard = latest.lock().unwrap();

            if let Some(frame) = guard.as_ref() {
                if frame.seq != last_seq {
                    send_frame(&mut ws, frame)?;
                    last_seq = frame.seq;
                }
            }
        }

        //
        // Send simulation time when it changes.
        //
        let time = *last_time.lock().unwrap();

        if time != last_sent_time {
            send_state(&mut ws, time)?;
            last_sent_time = time;
        }

        //
        // Avoid spinning at 100% CPU when there is nothing to do.
        //
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// WebSocket accept loop.
///
/// This listener has no HTTP responsibilities. Every connection received
/// here is expected to be a WebSocket connection.
fn websocket_server(
    listener: TcpListener,
    latest: &'static Mutex<Option<Frame>>,
    last_time: &'static Mutex<f64>,
) {
    println!("WebSocket server listening on ws://{HOST}:{WS_PORT}");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let latest = latest;
                let last_time = last_time;

                std::thread::spawn(move || {
                    if let Err(e) = serve_client(stream, latest, last_time) {
                        eprintln!("WebSocket connection error: {e}");
                    }
                });
            }

            Err(e) => {
                eprintln!("WebSocket accept error: {e}");
            }
        }
    }
}

/// Send a simple HTTP response.
fn send_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );

    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;

    Ok(())
}

/// Handle one ordinary HTTP connection.
fn handle_http_connection(mut stream: TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    //
    // This is intentionally a very small HTTP parser because the HTTP
    // server only needs to serve the embedded viewer.
    //
    let mut buffer = [0u8; 8192];

    let n = std::io::Read::read(&mut stream, &mut buffer)?;

    if n == 0 {
        return Ok(());
    }

    let request = std::str::from_utf8(&buffer[..n])?;

    let request_line = request.lines().next().unwrap_or("");

    let mut parts = request_line.split_whitespace();

    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    if method == "GET" && (path == "/" || path == "/index.html") {
        send_http_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            VIEWER_HTML.as_bytes(),
        )?;

        return Ok(());
    }

    send_http_response(
        &mut stream,
        "404 Not Found",
        "text/plain; charset=utf-8",
        b"Not Found",
    )?;

    Ok(())
}

/// Ordinary HTTP server for the browser viewer.
fn http_server(listener: TcpListener) {
    println!("HTTP server listening on http://{HOST}:{HTTP_PORT}");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                std::thread::spawn(move || {
                    if let Err(e) = handle_http_connection(stream) {
                        eprintln!("HTTP connection error: {e}");
                    }
                });
            }

            Err(e) => {
                eprintln!("HTTP accept error: {e}");
            }
        }
    }
}

pub fn run_server() {
    //
    // Load MuJoCo model.
    //
    let model = match std::env::args().nth(1) {
        Some(path) => {
            println!("loading MuJoCo model: {path}");
            MjModel::from_xml(&path)
        }

        None => {
            eprintln!("no model path given, using built-in demo model");

            MjModel::from_xml_string(DEMO_MODEL)
        }
    }
    .expect("failed to load model");

    //
    // Simulation state.
    //
    let mut data = MjData::new(&model);
    // make the ball bounce
    data.qvel_mut()[2] = 3.0;
    data.forward_kinematics();

    //
    // Headless renderer.
    //
    //
    // MjRenderer uses MuJoCo's offscreen rendering path, so this does
    // not require a visible window/display.
    //
    let mut renderer = MjRenderer::builder()
        .width(WIDTH)
        .height(HEIGHT)
        .rgb(true)
        .depth(false)
        .build(&model)
        .expect("failed to initialize headless renderer");

    //
    // Shared latest-frame state.
    //
    //
    // The simulation thread produces frames.
    // WebSocket client threads consume the newest frame.
    //
    let latest: &'static Mutex<Option<Frame>> = Box::leak(Box::new(Mutex::new(None)));

    let last_time: &'static Mutex<f64> = Box::leak(Box::new(Mutex::new(0.0)));

    //
    // HTTP listener.
    //
    let http_listener = TcpListener::bind((HOST, HTTP_PORT)).expect("failed to bind HTTP listener");

    //
    // WebSocket listener.
    //
    let ws_listener =
        TcpListener::bind((HOST, WS_PORT)).expect("failed to bind WebSocket listener");

    //
    // Start HTTP server.
    //
    {
        let listener = http_listener;

        std::thread::spawn(move || {
            http_server(listener);
        });
    }

    //
    // Start WebSocket server.
    //
    {
        let listener = ws_listener;

        std::thread::spawn(move || {
            websocket_server(listener, latest, last_time);
        });
    }

    println!();
    println!("Viewer:    http://{HOST}:{HTTP_PORT}");
    println!("WebSocket: ws://{HOST}:{WS_PORT}");
    println!();

    //
    // Simulation timing.
    //
    let timestep = model.opt().timestep;

    if timestep <= 0.0 {
        panic!("MuJoCo model has invalid timestep: {timestep}");
    }

    let frame_interval = Duration::from_secs_f64(1.0 / RENDER_FPS);

    let mut last = Instant::now();
    let mut sim_debt = 0.0f64;
    let mut seq = 0u64;

    //
    // Main simulation/render loop.
    //
    loop {
        let frame_start = Instant::now();

        let elapsed = frame_start.duration_since(last).as_secs_f64();

        last = frame_start;

        //
        // Accumulate wall-clock time that the simulation needs
        // to advance.
        //
        sim_debt += elapsed;

        let mut steps = (sim_debt / timestep) as u32;

        if steps > MAX_SIM_STEPS_PER_FRAME {
            steps = MAX_SIM_STEPS_PER_FRAME;
        }

        sim_debt -= (steps as f64) * timestep;

        //
        // Prevent an extended stall from causing an endless
        // simulation catch-up loop.
        //
        if sim_debt > timestep * 8.0 {
            sim_debt = 0.0;
        }

        //
        // Advance MuJoCo.
        //
        for _ in 0..steps {
            data.step();

            // Bounce when the ball reaches the floor.
            if data.qpos()[2] < 0.16 && data.qvel()[2] < 0.0 {
                data.qvel_mut()[2] = 5.0;
                data.forward_kinematics();
            }
        }

        // Synchronize simulation state with renderer and render
        // into the offscreen framebuffer.
        //
        match renderer
            .sync_data(&mut data)
            .and_then(|_| renderer.render())
        {
            Ok(()) => {
                if let Some(rgb) = renderer.rgb_flat() {
                    match encode_jpeg(rgb, WIDTH, HEIGHT) {
                        Ok(jpeg) => {
                            seq += 1;

                            let frame = Frame {
                                seq,
                                width: WIDTH as u16,
                                height: HEIGHT as u16,
                                jpeg,
                            };

                            //
                            // Replace the old frame.
                            //
                            // Clients always receive the newest
                            // frame rather than an ever-growing
                            // queue.
                            //
                            *latest.lock().unwrap() = Some(frame);

                            *last_time.lock().unwrap() = data.time();
                        }

                        Err(e) => {
                            eprintln!("JPEG encode error: {e}");
                        }
                    }
                }
            }

            Err(e) => {
                eprintln!("MuJoCo render error: {e}");
            }
        }

        //
        // Maintain the requested render rate.
        //
        let render_elapsed = frame_start.elapsed();

        if render_elapsed < frame_interval {
            std::thread::sleep(frame_interval - render_elapsed);
        }
    }
}

