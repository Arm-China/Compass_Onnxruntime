# Environment variable description

#### OPERATOR_PATH

The path of the operator library.
The default value is './operator'.
Usage : export OPERATOR_PATH=/path/to/Compass_Runtime_onnxruntime/operator

#### ENABLE_PROFILE

Enable the profile function, which is a performance data inside zhouyi.

* on
* off

The default value is 'off'.
Usage : export ENABLE_PROFILE=on
When this function is turned on, the actual running speed will be seriously affected.
When this function is turned on, The result of X3/X3P will be invalid.

#### PROFILE_REPORT_SUFFIX

Set the format of the profile perf file.
Support value:

* csv
* html
* json

Usage : export PROFILE_REPORT_SUFFIX=csv

#### INTERMIDIATE_PATH

The path where the intermediate files and dump files are stored during the compilation process.
The path must have read and write permissions.
The default value is './'.
Usage : export INTERMIDIATE_PATH=/path

#### DUMP_FILE

Enable file dump function. It will generate graph and other files.

* on
* off

The default value is 'off'.
Usage : export DUMP_FILE=on

#### TOOLKIT_LOG_LEVEL

This corresponds to the log level in the the zhouyinpu-toolkit.

* 0(defalut): LOG_ERROR
* 1: LOG_WARN
* 2: LOG_INFO
* 3: LOG_DEBUG
* 4: LOG_VERBOSE

Usage : export TOOLKIT_LOG_LEVEL=4

#### TILING_METHOD

This corresponds to the opening of the aipu runtime tiling function.

* fps: pay close attention to fps.
* The default value is 'fps'.
* none: disable tiling.

Usage : export TILING_METHOD=fps

#### SIMULATOR_ARCH

This corresponds to the switching of the aipuruntime simulator arch.
Support:

* X2_1204MP3
* X3P_1304
* X3P_1304MP2
* X3P_1304MP4
* X3S_1304
* X3S_1304MP2
* X3S_1304MP4

Usage : export SIMULATOR_ARCH=X2_1204MP3
