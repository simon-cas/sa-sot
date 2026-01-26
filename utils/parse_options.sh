#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# This script is used to parse command line arguments in bash scripts.
# Usage: . utils/parse_options.sh || exit 1

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options can be specified as:"
    echo "  --option=value"
    echo "  --option value"
    echo ""
    echo "Available options:"
    echo "  --stage=N              Start from stage N"
    echo "  --stop_stage=N        Stop at stage N"
    echo "  --data=DIR            Data directory"
    echo "  --exp=DIR             Experiment directory"
    echo "  --exp_name=NAME       Experiment name"
    echo "  --gpus=GPUS           GPU IDs (space-separated)"
    echo "  --config=FILE         Config file path"
    echo "  --skip-alignment      Skip forced alignment"
    echo "  --skip-eval           Skip evaluation dataset preparation"
    echo "  --single-speaker      Train with single speaker data only"
    echo "  --reset-training      Reset training: remove old checkpoints and start fresh"
    echo ""
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --*=*)
            # Handle --option=value format
            optarg=$(echo "$1" | sed 's/[-_a-zA-Z0-9]*=//')
            optname=$(echo "$1" | sed 's/=.*//')
            optname=$(echo "$optname" | sed 's/--//')
            eval "$optname=\"$optarg\""
            shift
            ;;
        --skip-alignment)
            skip_alignment=true
            shift
            ;;
        --skip-eval)
            skip_eval=true
            shift
            ;;
        --single-speaker)
            single_speaker=true
            shift
            ;;
        --reset-training)
            reset_training=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # Handle --option value format
            optname=$(echo "$1" | sed 's/--//')
            shift
            if [ $# -eq 0 ]; then
                echo "Error: Option --$optname requires a value"
                exit 1
            fi
            eval "$optname=\"$1\""
            shift
            ;;
    esac
done

