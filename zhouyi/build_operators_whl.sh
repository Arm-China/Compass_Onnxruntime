#!/bin/sh -e

build_help() {
    echo "====================build operator wheel package============================"
    echo "Build Options:"
    echo "-h, --help                    help"
    echo "-a, --arch                    Zhouyi arch (default x3p):"
    echo "                              - x3p"
    echo "                              - x2"
    echo "-k, --toolkit                 The aipu toolkit path"
    echo "  ./build_operators_whl.sh -a x3p -k /project/ai/scratch01/Compass_Runtime_Midware/share/x3p"
    echo "  ./build_operators_whl.sh -a x2 -k /project/ai/scratch01/Compass_Runtime_Midware/share/x2"
    echo "==========================================================================="
    exit 1
}

ZHOUYI_ARCH=x3p
AIPU_TOOLKIL_HOME=/project/ai/scratch01/Compass_Runtime_Midware/share/x3p

while [ -n "$1" ]
do
    case "$1" in
    -h|--help)
        build_help
        ;;
    -a|--arch)
        ZHOUYI_ARCH="$2"
        shift
        ;;
    -k|--toolkit)
        AIPU_TOOLKIL_HOME="$2"
        shift
        ;;
    --)
         shift;
         break
         ;;
     *)
         break
    esac
    shift
done

LOCAL_PATH=$(cd "$(dirname "$0")";pwd)
PROJ_DIR=$LOCAL_PATH
if [ -d $PROJ_DIR/operators/dist ]; then
    rm -rf $PROJ_DIR/operators/dist
fi

# operator
local_operator_path=$PROJ_DIR/operators/ZhouyiOperators/operators
if [ ! -d $local_operator_path ]; then
    mkdir -p $local_operator_path
fi
cp -f $AIPU_TOOLKIL_HOME/operator/* $local_operator_path

if [ ! -f $PROJ_DIR/operators/setup.py ]; then
    echo "Cannot find setup.py!"
    exit 1
fi
#
pushd $PROJ_DIR/operators
python3 setup.py clean --all
python3 setup.py bdist_wheel --arch=$ZHOUYI_ARCH
popd
