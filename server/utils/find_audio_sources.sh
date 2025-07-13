#!/bin/bash

# Audio Source Finder
# This script helps you find the correct source IDs for creating virtual audio devices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_highlight() {
    echo -e "${CYAN}$1${NC}"
}

# Function to check if PulseAudio is available
check_pulseaudio() {
    if ! command -v pactl &> /dev/null; then
        print_error "pactl not found. PulseAudio is required."
        exit 1
    fi
    
    if ! pactl info &> /dev/null; then
        print_error "PulseAudio is not running. Please start PulseAudio first."
        exit 1
    fi
}

# Function to list all sources with detailed information
list_all_sources() {
    print_highlight "=== ALL AUDIO SOURCES ==="
    echo ""
    
    pactl list short sources | while read -r line; do
        if [[ -n "$line" ]]; then
            source_id=$(echo "$line" | cut -f1)
            source_name=$(echo "$line" | cut -f2)
            source_desc=$(echo "$line" | cut -f3)
            
            echo -e "${GREEN}ID: $source_id${NC}"
            echo "  Name: $source_name"
            echo "  Description: $source_desc"
            echo ""
        fi
    done
}

# Function to list only input sources (hardware microphones)
list_input_sources() {
    print_highlight "=== HARDWARE INPUT SOURCES (Microphones) ==="
    echo ""
    
    local found_inputs=false
    
    while read -r line; do
        if [[ -n "$line" ]]; then
            source_id=$(echo "$line" | cut -f1)
            source_name=$(echo "$line" | cut -f2)
            source_desc=$(echo "$line" | cut -f3)
            
            # Look for hardware input sources (not monitors or virtual devices)
            if [[ "$source_name" != *".monitor"* ]] && \
               [[ "$source_name" != *"Virtual"* ]] && \
               [[ "$source_name" != *"null"* ]] && \
               [[ "$source_desc" != *"Monitor"* ]]; then
                
                echo -e "${GREEN}ID: $source_id${NC}"
                echo "  Name: $source_name"
                echo "  Description: $source_desc"
                echo ""
                found_inputs=true
            fi
        fi
    done < <(pactl list short sources)
    
    if [[ "$found_inputs" == false ]]; then
        print_warning "No hardware input sources found"
    fi
}

# Function to list monitor sources (output monitoring)
list_monitor_sources() {
    print_highlight "=== MONITOR SOURCES (Output Monitoring) ==="
    echo ""
    
    local found_monitors=false
    
    pactl list short sources | while read -r line; do
        if [[ -n "$line" ]]; then
            source_id=$(echo "$line" | cut -f1)
            source_name=$(echo "$line" | cut -f2)
            source_desc=$(echo "$line" | cut -f3)
            
            # Look for monitor sources
            if [[ "$source_name" == *".monitor"* ]] || \
               [[ "$source_desc" == *"Monitor"* ]]; then
                
                echo -e "${GREEN}ID: $source_id${NC}"
                echo "  Name: $source_name"
                echo "  Description: $source_desc"
                echo ""
                found_monitors=true
            fi
        fi
    done
    
    if [[ "$found_monitors" == false ]]; then
        print_warning "No monitor sources found"
    fi
}

# Function to show usage
show_usage() {
    echo "Audio Source Finder"
    echo "==================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --all               List all audio sources"
    echo "  -i, --inputs            List only hardware input sources (default)"
    echo "  -m, --monitors          List only monitor sources"
    echo "  -c, --create-example    Show example command for creating virtual device"
    echo ""
    echo "Examples:"
    echo "  $0                      # List hardware input sources"
    echo "  $0 -a                   # List all sources"
    echo "  $0 -m                   # List monitor sources"
    echo "  $0 -c                   # Show example creation command"
    echo ""
    echo "To create a virtual device, use:"
    echo "  ../create_virtual_audio_device.sh <source1_id> <source2_id>"
}

# Function to show example creation command
show_example_command() {
    print_highlight "=== EXAMPLE VIRTUAL DEVICE CREATION ==="
    echo ""
    
    # Get first two input sources
    local source1_id=""
    local source2_id=""
    
    while read -r line; do
        if [[ -n "$line" ]]; then
            source_id=$(echo "$line" | cut -f1)
            source_name=$(echo "$line" | cut -f2)
            
            # Look for hardware input sources
            if [[ "$source_name" != *".monitor"* ]] && \
               [[ "$source_name" != *"Virtual"* ]] && \
               [[ "$source_name" != *"null"* ]]; then
                
                if [[ -z "$source1_id" ]]; then
                    source1_id="$source_id"
                elif [[ -z "$source2_id" ]]; then
                    source2_id="$source_id"
                    break
                fi
            fi
        fi
    done < <(pactl list short sources)
    
    if [[ -n "$source1_id" && -n "$source2_id" ]]; then
        echo "Found two input sources:"
        echo "  Source 1 ID: $source1_id"
        echo "  Source 2 ID: $source2_id"
        echo ""
        echo "To create a virtual device combining these sources, run:"
        echo -e "${GREEN}../create_virtual_audio_device.sh $source1_id $source2_id${NC}"
        echo ""
        echo "Or with a custom name:"
        echo -e "${GREEN}../create_virtual_audio_device.sh -n MyVirtualMix $source1_id $source2_id${NC}"
    else
        print_warning "Could not find two input sources for example"
        echo "Please list sources first with: $0 -a"
    fi
}

# Parse command line arguments
SHOW_ALL=false
SHOW_INPUTS=true
SHOW_MONITORS=false
SHOW_EXAMPLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -a|--all)
            SHOW_ALL=true
            SHOW_INPUTS=false
            SHOW_MONITORS=false
            shift
            ;;
        -i|--inputs)
            SHOW_INPUTS=true
            SHOW_ALL=false
            SHOW_MONITORS=false
            shift
            ;;
        -m|--monitors)
            SHOW_MONITORS=true
            SHOW_ALL=false
            SHOW_INPUTS=false
            shift
            ;;
        -c|--create-example)
            SHOW_EXAMPLE=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Main execution
main() {
    print_info "Audio Source Finder"
    echo ""
    
    # Check PulseAudio availability
    check_pulseaudio
    
    # Show requested information
    if [[ "$SHOW_ALL" == true ]]; then
        list_all_sources
    elif [[ "$SHOW_MONITORS" == true ]]; then
        list_monitor_sources
    else
        list_input_sources
    fi
    
    if [[ "$SHOW_EXAMPLE" == true ]]; then
        echo ""
        show_example_command
    fi
    
    echo ""
    print_info "To create a virtual device, use the create_virtual_audio_device.sh script"
    print_info "Example: ../create_virtual_audio_device.sh 0 1"
}

# Run main function
main "$@" 