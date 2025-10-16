#!/bin/bash
script_dir=$(cd $(dirname $0);pwd)

thread_num=32

build_help() {
  echo "============================example Build Help=============================="
  echo "Build Options:"
  echo "-h, --help                    help"
  echo "-c, --compiler-path           The Android NDK path/Linux GCC path"
  echo "-n, --onnxruntime             The onnxruntime code path"
  echo "-l, --lib                     The libonnxruntime.so libarary path"
  echo "--prefix                      Compiler prefix"
  echo "Usage:"
  echo " ./build.sh -c /arm/tools/gnu/gcc/9.3.0/rhe7-x86_64 -n /path/to/onnxruntime
          -l /path/to/libonnxruntime.so"
  echo " ./build.sh -c /path/to/gcc-arm-10.3-x86_64-aarch64-none-linux-gnu -n /path/to/onnxruntime
          -l /path/to/libonnxruntime.so  --prefix aarch64-none-linux-gnu-"
  echo "==========================================================================="
  exit 1
}

TARGET="x86_64"
COMPILER_PATH=
COMPILER_PREFIX=

while [ -n "$1" ]
do
    case "$1" in
    -h|--help)
        build_help
        ;;
    -c|--compiler-path)
        COMPILER_PATH="$2"
        shift
        ;;
    --prefix)
        COMPILER_PREFIX="$2"
        shift
        ;;
     -n|--onnxruntime)
        ONNXRUNTIME_ROOTDIR="$2"
        shift
        ;;
     -l|--lib)
        ONNXRUNTIME_LIB_DIR="$2"
        shift
        ;;
     --)
         shift ;
         break
         ;;
     *)
         break
    esac
    shift
done


echo $COMPILER_PATH

mkdir -p $script_dir/build  && cd $script_dir/build

build_flag=" -DCMAKE_C_COMPILER=$COMPILER_PATH/bin/${COMPILER_PREFIX}gcc \
            -DCMAKE_CXX_COMPILER=$COMPILER_PATH/bin/${COMPILER_PREFIX}g++ \
            -Donnxruntime_USE_ZHOUYI=ON "

target_flag=$build_flag" -DONNXRUNTIME_ROOTDIR=$ONNXRUNTIME_ROOTDIR \
            -DONNXRUNTIME_LIB_DIR=$ONNXRUNTIME_LIB_DIR "
echo "cmake $target_flag$script_dir"|sed 's/[\t ]\+/\n/g'
cd $script_dir/build
cmake $target_flag $script_dir
make -j${thread_num}
