#!/bin/bash

# Virtual Audio Device Creator
# This script creates a virtual audio device that combines two hardware audio sources
# using PulseAudio's module-null-sink and module-loopback modules.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VIRTUAL_SINK_NAME="Virtual_Mixed_Audio"
SAMPLE_RATE="44100"
CHANNELS="2"

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

# Function to check if PulseAudio is available
check_pulseaudio() {
    if ! command -v pactl &> /dev/null; then
        print_error "pactl not found. PulseAudio is required for this script."
        exit 1
    fi
    
    if ! pactl info &> /dev/null; then
        print_error "PulseAudio is not running. Please start PulseAudio first."
        exit 1
    fi
    
    print_success "PulseAudio is available and running"
}

# Function to list available audio sources
list_sources() {
    print_info "Available audio sources:"
    echo "=================================="
    pactl list short sources | while read -r line; do
        if [[ -n "$line" ]]; then
            source_id=$(echo "$line" | cut -f1)
            source_name=$(echo "$line" | cut -f2)
            source_desc=$(echo "$line" | cut -f3)
            echo "ID: $source_id | Name: $source_name | Description: $source_desc"
        fi
    done
    echo ""
}

# Function to list existing sinks
list_sinks() {
    print_info "Existing audio sinks:"
    echo "=========================="
    pactl list short sinks | while read -r line; do
        if [[ -n "$line" ]]; then
            sink_id=$(echo "$line" | cut -f1)
            sink_name=$(echo "$line" | cut -f2)
            sink_desc=$(echo "$line" | cut -f3)
            echo "ID: $sink_id | Name: $sink_name | Description: $sink_desc"
        fi
    done
    echo ""
}

# Function to validate source ID
validate_source() {
    local source_id=$1
    if ! pactl list short sources | grep -q "^$source_id[[:space:]]"; then
        print_error "Source ID $source_id does not exist"
        return 1
    fi
    return 0
}

# Function to create virtual audio device
create_virtual_device() {
    local source1_id=$1
    local source2_id=$2
    local sink_name=$3
    
    print_info "Creating virtual audio device..."
    print_info "Source 1 ID: $source1_id"
    print_info "Source 2 ID: $source2_id"
    print_info "Virtual sink name: $sink_name"
    
    # Check if sink already exists
    if pactl list short sinks | grep -q "$sink_name"; then
        print_warning "Sink '$sink_name' already exists. Removing it first..."
        remove_virtual_device "$sink_name"
    fi
    
    # Create null sink (virtual device)
    print_info "Creating null sink..."
    local sink_module_id
    sink_module_id=$(pactl load-module module-null-sink \
        sink_name="$sink_name" \
        sink_properties=device.description="$sink_name" \
        rate="$SAMPLE_RATE" \
        channels="$CHANNELS")
    
    if [[ -z "$sink_module_id" ]]; then
        print_error "Failed to create null sink"
        exit 1
    fi
    
    print_success "Created null sink with module ID: $sink_module_id"
    
    # Get the monitor source for the null sink
    local monitor_source
    monitor_source=$(pactl list short sources | grep "$sink_name.monitor" | cut -f1)
    
    if [[ -z "$monitor_source" ]]; then
        print_error "Could not find monitor source for sink '$sink_name'"
        exit 1
    fi
    
    print_info "Monitor source ID: $monitor_source"
    
    # Create loopback from source 1 to the virtual sink
    print_info "Creating loopback from source 1..."
    local loopback1_module_id
    loopback1_module_id=$(pactl load-module module-loopback \
        source="$source1_id" \
        sink="$sink_name" \
        latency_msec=1)
    
    if [[ -z "$loopback1_module_id" ]]; then
        print_error "Failed to create loopback from source 1"
        exit 1
    fi
    
    print_success "Created loopback 1 with module ID: $loopback1_module_id"
    
    # Create loopback from source 2 to the virtual sink
    print_info "Creating loopback from source 2..."
    local loopback2_module_id
    loopback2_module_id=$(pactl load-module module-loopback \
        source="$source2_id" \
        sink="$sink_name" \
        latency_msec=1)
    
    if [[ -z "$loopback2_module_id" ]]; then
        print_error "Failed to create loopback from source 2"
        exit 1
    fi
    
    print_success "Created loopback 2 with module ID: $loopback2_module_id"
    
    # Save module IDs to a file for cleanup
    echo "$sink_module_id" > "/tmp/virtual_audio_${sink_name}_modules.txt"
    echo "$loopback1_module_id" >> "/tmp/virtual_audio_${sink_name}_modules.txt"
    echo "$loopback2_module_id" >> "/tmp/virtual_audio_${sink_name}_modules.txt"
    
    print_success "Virtual audio device '$sink_name' created successfully!"
    print_info "The virtual device is now available as an audio source in your system"
    print_info "You can use it in applications by selecting '$sink_name' as the input device"
    
    # Show the new sink
    print_info "New virtual sink details:"
    pactl list short sinks | grep "$sink_name"
}

# Function to remove virtual audio device
remove_virtual_device() {
    local sink_name=$1
    
    print_info "Removing virtual audio device '$sink_name'..."
    
    # Check if module IDs file exists
    local module_file="/tmp/virtual_audio_${sink_name}_modules.txt"
    if [[ -f "$module_file" ]]; then
        print_info "Unloading modules..."
        while read -r module_id; do
            if [[ -n "$module_id" ]]; then
                pactl unload-module "$module_id" 2>/dev/null || true
                print_info "Unloaded module: $module_id"
            fi
        done < "$module_file"
        rm -f "$module_file"
        print_success "Virtual audio device '$sink_name' removed successfully"
    else
        print_warning "No module tracking file found for '$sink_name'"
        print_info "Attempting to find and remove modules manually..."
        
        # Try to find and remove modules manually
        pactl list short modules | grep -i "$sink_name" | cut -f1 | while read -r module_id; do
            pactl unload-module "$module_id" 2>/dev/null || true
        done
    fi
}

# Function to show usage
show_usage() {
    echo "Virtual Audio Device Creator"
    echo "============================"
    echo ""
    echo "Usage: $0 [OPTIONS] <source1_id> <source2_id>"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -n, --name NAME         Virtual sink name (default: $VIRTUAL_SINK_NAME)"
    echo "  -r, --rate RATE         Sample rate (default: $SAMPLE_RATE)"
    echo "  -c, --channels CHANNELS Number of channels (default: $CHANNELS)"
    echo "  -l, --list              List available sources and sinks"
    echo "  -r, --remove NAME       Remove virtual device with given name"
    echo ""
    echo "Examples:"
    echo "  $0 0 1                    # Create virtual device from sources 0 and 1"
    echo "  $0 -n MyMix 2 3           # Create virtual device named 'MyMix' from sources 2 and 3"
    echo "  $0 -r MyMix               # Remove virtual device named 'MyMix'"
    echo "  $0 -l                     # List available sources and sinks"
    echo ""
    echo "To find source IDs, use: pactl list short sources"
}

# Function to handle cleanup on script exit
cleanup() {
    print_info "Cleaning up..."
    # Remove any temporary files
    rm -f /tmp/virtual_audio_*_modules.txt
}

# Set up trap for cleanup
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -n|--name)
            VIRTUAL_SINK_NAME="$2"
            shift 2
            ;;
        -r|--rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        -c|--channels)
            CHANNELS="$2"
            shift 2
            ;;
        -l|--list)
            check_pulseaudio
            list_sources
            list_sinks
            exit 0
            ;;
        --remove)
            check_pulseaudio
            remove_virtual_device "$2"
            exit 0
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

# Check if we have the required arguments
if [[ $# -ne 2 ]]; then
    print_error "Invalid number of arguments"
    show_usage
    exit 1
fi

SOURCE1_ID=$1
SOURCE2_ID=$2

# Main execution
main() {
    print_info "Starting virtual audio device creation..."
    
    # Check PulseAudio availability
    check_pulseaudio
    
    # Validate source IDs
    if ! validate_source "$SOURCE1_ID"; then
        exit 1
    fi
    
    if ! validate_source "$SOURCE2_ID"; then
        exit 1
    fi
    
    if [[ "$SOURCE1_ID" == "$SOURCE2_ID" ]]; then
        print_error "Source 1 and Source 2 cannot be the same"
        exit 1
    fi
    
    # Show current state
    print_info "Current audio sources:"
    list_sources
    
    print_info "Current audio sinks:"
    list_sinks
    
    # Create the virtual device
    create_virtual_device "$SOURCE1_ID" "$SOURCE2_ID" "$VIRTUAL_SINK_NAME"
    
    # Show final state
    print_info "Updated audio sinks:"
    list_sinks
    
    print_success "Virtual audio device creation completed!"
    print_info "To remove the virtual device, run: $0 --remove $VIRTUAL_SINK_NAME"
}

# Run main function
main "$@" 