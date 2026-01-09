#!/bin/bash

# ============================================================
# Project Sync Script - play2048
# ============================================================
# Syncs project to remote server, excluding development files
#
# Usage:
#   ./sync.sh                    # Interactive mode
#   ./sync.sh --dry-run          # Test without uploading
#   ./sync.sh user@host:/path    # Direct sync
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Configuration
# ============================================================

# Default remote (change this to your server)
DEFAULT_REMOTE="user@server:/path/to/play2048"

# Exclusion patterns
EXCLUDE_PATTERNS=(
    # Version control
    ".git"
    ".gitignore"
    ".github"
    
    # Python cache
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".Python"
    "*.so"
    
    # Virtual environments
    "venv"
    "env"
    ".venv"
    ".env"
    "ENV"
    
    # IDE and editor files
    ".vscode"
    ".idea"
    "*.swp"
    "*.swo"
    "*~"
    ".DS_Store"
    
    # Development outputs
    "checkpoints"
    "logs"
    "results/training_curves"
    "results/evaluation"
    
    # TensorBoard
    "**/tensorboard"
    "runs"
    "*.tfevents.*"
    
    # Test and temp files
    "test_*"
    "tmp"
    "temp"
    "*.tmp"
    "*.log"
    
    # Data files (optional - uncomment if too large)
    # "data/training_games.jsonl"
    # "data/augmented"
    # "data/processed"
    
    # Node modules (if any)
    "node_modules"
    
    # Build artifacts
    "build"
    "dist"
    "*.egg-info"
    
    # OS files
    "Thumbs.db"
    "desktop.ini"
    
    # Package manager
    ".uv"
    "uv.lock"
)

# ============================================================
# Functions
# ============================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Build exclusion arguments for rsync
build_exclude_args() {
    local args=""
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        args="$args --exclude='$pattern'"
    done
    echo "$args"
}

# Show what will be synced
show_sync_preview() {
    local remote=$1
    local dry_run=${2:-false}
    
    print_header "Sync Preview"
    echo "Source: $(pwd)"
    echo "Destination: $remote"
    echo ""
    
    if [ "$dry_run" = true ]; then
        print_info "Running in DRY-RUN mode (no files will be transferred)"
    fi
    
    echo ""
    echo "Excluded patterns:"
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        echo "  - $pattern"
    done
    echo ""
}

# Perform the sync
do_sync() {
    local remote=$1
    local dry_run=${2:-false}
    
    # Build rsync command
    local rsync_cmd="rsync -avz --progress --delete"
    
    # Add dry-run flag if requested
    if [ "$dry_run" = true ]; then
        rsync_cmd="$rsync_cmd --dry-run"
    fi
    
    # Add exclusions
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        rsync_cmd="$rsync_cmd --exclude='$pattern'"
    done
    
    # Add source and destination
    rsync_cmd="$rsync_cmd ./ $remote"
    
    # Show command
    print_info "Running: rsync -avz --progress --delete [with exclusions] ./ $remote"
    echo ""
    
    # Execute
    eval "$rsync_cmd"
    
    # Check result
    if [ $? -eq 0 ]; then
        echo ""
        if [ "$dry_run" = true ]; then
            print_success "Dry-run completed successfully!"
            print_info "Remove --dry-run to perform actual sync"
        else
            print_success "Sync completed successfully!"
        fi
    else
        print_error "Sync failed!"
        exit 1
    fi
}

# Interactive mode
interactive_mode() {
    print_header "Interactive Sync Setup"
    
    echo ""
    echo "Enter remote destination (or press Enter for default):"
    echo "Format: user@host:/path/to/destination"
    echo "Default: $DEFAULT_REMOTE"
    echo ""
    read -p "Remote: " remote
    
    if [ -z "$remote" ]; then
        remote=$DEFAULT_REMOTE
    fi
    
    echo ""
    echo "Perform dry-run first? (recommended) [Y/n]"
    read -p "Dry-run: " dry_run_choice
    
    case "$dry_run_choice" in
        [Nn]*)
            dry_run=false
            ;;
        *)
            dry_run=true
            ;;
    esac
    
    echo ""
    show_sync_preview "$remote" "$dry_run"
    
    echo ""
    read -p "Proceed with sync? [y/N] " confirm
    
    case "$confirm" in
        [Yy]*)
            do_sync "$remote" "$dry_run"
            
            # If dry-run, ask to do real sync
            if [ "$dry_run" = true ]; then
                echo ""
                read -p "Perform actual sync now? [y/N] " confirm_real
                case "$confirm_real" in
                    [Yy]*)
                        echo ""
                        do_sync "$remote" false
                        ;;
                    *)
                        print_info "Sync cancelled"
                        ;;
                esac
            fi
            ;;
        *)
            print_info "Sync cancelled"
            ;;
    esac
}

# ============================================================
# Main
# ============================================================

print_header "play2048 - Remote Sync Tool"

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    print_error "rsync is not installed!"
    echo "Install with: brew install rsync (macOS) or apt-get install rsync (Linux)"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments - interactive mode
    interactive_mode
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    # Show help
    echo "Usage:"
    echo "  ./sync.sh                    Interactive mode"
    echo "  ./sync.sh --dry-run          Dry-run with default remote"
    echo "  ./sync.sh user@host:/path    Sync to specified remote"
    echo "  ./sync.sh --dry-run user@host:/path"
    echo ""
    echo "Options:"
    echo "  --dry-run    Test sync without transferring files"
    echo "  --help       Show this help message"
    echo ""
    echo "Configuration:"
    echo "  Default remote: $DEFAULT_REMOTE"
    echo "  Edit this script to change default settings"
elif [ "$1" = "--dry-run" ]; then
    # Dry-run mode
    if [ -z "$2" ]; then
        remote=$DEFAULT_REMOTE
    else
        remote=$2
    fi
    show_sync_preview "$remote" true
    echo ""
    do_sync "$remote" true
else
    # Direct sync to specified remote
    remote=$1
    dry_run=false
    
    # Check if --dry-run is in arguments
    if [[ "$*" == *"--dry-run"* ]]; then
        dry_run=true
    fi
    
    show_sync_preview "$remote" "$dry_run"
    echo ""
    read -p "Proceed with sync? [y/N] " confirm
    
    case "$confirm" in
        [Yy]*)
            do_sync "$remote" "$dry_run"
            ;;
        *)
            print_info "Sync cancelled"
            ;;
    esac
fi

echo ""
print_header "Sync Complete"
