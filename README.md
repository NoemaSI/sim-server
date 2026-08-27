# Setup MacOS

1) download mujoco release 3.9.0 from https://github.com/google-deepmind/mujoco/releases#release-3.9.0
2) extract the DMG to a folder or just copy the files after opening the DMG
3) copy (from DMG content)  `mujoco.framework/Versions/Current/libmujoco.3.9.0.dylib` into this folder as `libmujoco.dylib`
4) copy `mujoco.framework/Versions/Current/libmujoco.3.9.0.dylib` into `~/lib/` (dont rename it, create the ~/lib folder if it doesn't exist)
4) export MUJOCO_DYNAMIC_LINK_DIR=/PATH/TO/THIS/FOLDER

> Please create an issue if you have trouble getting things to run on MacOS
