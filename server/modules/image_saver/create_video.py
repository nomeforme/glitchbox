#!/usr/bin/env python3
"""
Video Creator for Image Saver Module

This script converts saved images from a session folder into a video file.
It can automatically find the latest session or work with a specified folder.
"""

import os
import argparse
import glob
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class VideoCreator:
    """Creates videos from saved image sequences."""
    
    def __init__(
        self,
        base_dir: str = "output",
        fps: int = 10,
        video_format: str = "mp4",
        video_quality: int = 23,  # CRF value for H.264 (lower = better quality)
        compress: bool = False,
        compression_preset: str = "slow",
        compression_crf: int = 23,
        audio_bitrate: str = "128k",
        debug: bool = False
    ):
        """
        Initialize the video creator.
        
        Args:
            base_dir: Base directory containing session folders
            fps: Frames per second for the output video
            video_format: Video format (mp4, avi, mov)
            video_quality: Video quality (CRF value for H.264, 0-51, lower = better)
            compress: Whether to apply additional ffmpeg compression
            compression_preset: FFmpeg preset for compression (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            compression_crf: CRF value for ffmpeg compression (0-51, lower = better quality)
            audio_bitrate: Audio bitrate for compression (e.g., "128k", "192k", "256k")
            debug: Enable debug output
        """
        self.base_dir = os.path.abspath(base_dir)
        self.fps = fps
        self.video_format = video_format.lower()
        self.video_quality = video_quality
        self.compress = compress
        self.compression_preset = compression_preset
        self.compression_crf = compression_crf
        self.audio_bitrate = audio_bitrate
        self.debug = debug
        
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        if self.debug:
            print(f"[VideoCreator] Base directory: {self.base_dir}")
            print(f"[VideoCreator] FPS: {self.fps}")
            print(f"[VideoCreator] Format: {self.video_format}")
            print(f"[VideoCreator] Compression enabled: {self.compress}")
            if self.compress:
                print(f"[VideoCreator] Compression preset: {self.compression_preset}")
                print(f"[VideoCreator] Compression CRF: {self.compression_crf}")
    
    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available in the system."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _compress_video_with_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """
        Compress video using ffmpeg with H.264 encoding.
        
        Args:
            input_path: Path to input video file
            output_path: Path for compressed output video
            
        Returns:
            True if compression successful, False otherwise
        """
        if not self._check_ffmpeg_available():
            print("Warning: ffmpeg not found. Skipping compression.")
            return False
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', self.compression_preset,
            '-crf', str(self.compression_crf),
            '-c:a', 'aac',
            '-b:a', self.audio_bitrate,
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        if self.debug:
            print(f"[VideoCreator] Running ffmpeg compression:")
            print(f"[VideoCreator] Command: {' '.join(cmd)}")
        
        try:
            print(f"Compressing video with ffmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if self.debug:
                    print(f"[VideoCreator] FFmpeg compression successful")
                return True
            else:
                print(f"FFmpeg compression failed:")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("FFmpeg compression timed out (5 minutes)")
            return False
        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            return False
    
    def find_latest_session(self) -> Optional[str]:
        """Find the latest session folder in the base directory."""
        if not os.path.exists(self.base_dir):
            print(f"Error: Base directory does not exist: {self.base_dir}")
            return None
        
        # Look for session folders matching the pattern session_YYYYMMDD_HHMMSS
        session_pattern = os.path.join(self.base_dir, "session_*")
        session_folders = glob.glob(session_pattern)
        
        if not session_folders:
            print(f"No session folders found in {self.base_dir}")
            return None
        
        # Sort by modification time (most recent first)
        session_folders.sort(key=os.path.getmtime, reverse=True)
        latest_session = session_folders[0]
        
        if self.debug:
            print(f"[VideoCreator] Found {len(session_folders)} session folders")
            print(f"[VideoCreator] Latest session: {latest_session}")
        
        return latest_session
    
    def get_image_files(self, session_dir: str) -> List[str]:
        """Get all image files from the session directory, sorted by filename."""
        if not os.path.exists(session_dir):
            print(f"Error: Session directory does not exist: {session_dir}")
            return []
        
        image_files = []
        for ext in self.image_extensions:
            pattern = os.path.join(session_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            print(f"No image files found in {session_dir}")
            return []
        
        # Sort by filename to maintain chronological order
        # The filename format is image_XXXXXX_HHMMSS_mmm.ext
        def extract_sort_key(filepath):
            filename = os.path.basename(filepath)
            # Extract the counter and timestamp for sorting
            match = re.match(r'image_(\d+)_(\d+)_(\d+)', filename)
            if match:
                counter = int(match.group(1))
                timestamp = int(match.group(2))
                milliseconds = int(match.group(3))
                return (counter, timestamp, milliseconds)
            return (0, 0, 0)
        
        image_files.sort(key=extract_sort_key)
        
        if self.debug:
            print(f"[VideoCreator] Found {len(image_files)} image files")
            print(f"[VideoCreator] First image: {os.path.basename(image_files[0])}")
            print(f"[VideoCreator] Last image: {os.path.basename(image_files[-1])}")
        
        return image_files
    
    def get_video_dimensions(self, image_files: List[str]) -> Tuple[int, int]:
        """Get the dimensions for the video based on the first image."""
        if not image_files:
            return (512, 512)  # Default dimensions
        
        try:
            with Image.open(image_files[0]) as img:
                width, height = img.size
                if self.debug:
                    print(f"[VideoCreator] Video dimensions: {width}x{height}")
                return width, height
        except Exception as e:
            print(f"Error reading image dimensions: {e}")
            return (512, 512)
    
    def create_video(
        self, 
        session_dir: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a video from images in the specified session directory.
        
        Args:
            session_dir: Path to session directory (if None, uses latest)
            output_filename: Custom output filename (if None, auto-generates)
            
        Returns:
            Path to created video file, or None if failed
        """
        # Find session directory
        if session_dir is None:
            session_dir = self.find_latest_session()
            if session_dir is None:
                return None
        
        if not os.path.exists(session_dir):
            print(f"Error: Session directory does not exist: {session_dir}")
            return None
        
        # Get image files
        image_files = self.get_image_files(session_dir)
        if not image_files:
            return None
        
        # Create videos subdirectory
        videos_dir = os.path.join(session_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Generate output filename
        if output_filename is None:
            session_name = os.path.basename(session_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{session_name}_video_{timestamp}"
            output_filename = f"{base_filename}.{self.video_format}"
        else:
            # Extract base filename without extension for compression
            base_filename = os.path.splitext(output_filename)[0]
        
        output_path = os.path.join(videos_dir, output_filename)
        
        # Get video dimensions
        width, height = self.get_video_dimensions(image_files)
        
        # Set up video writer
        fourcc = self._get_fourcc()
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return None
        
        print(f"Creating video: {output_path}")
        print(f"Processing {len(image_files)} images at {self.fps} FPS...")
        
        # Process images
        for i, image_path in enumerate(image_files):
            try:
                # Load image with PIL to handle different formats
                with Image.open(image_path) as pil_img:
                    # Convert to RGB if necessary
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    # Resize if necessary
                    if pil_img.size != (width, height):
                        pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # Convert PIL to OpenCV format (BGR)
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    video_writer.write(cv_img)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                    progress = (i + 1) / len(image_files) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(image_files)})")
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        # Clean up
        video_writer.release()
        
        # Verify the video was created
        if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
            print(f"Error: Video creation failed")
            return None
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        duration = len(image_files) / self.fps
        print(f"Initial video created successfully!")
        print(f"Output: {output_path}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Frames: {len(image_files)}")
        
        # Apply ffmpeg compression if requested
        final_output_path = output_path
        if self.compress:
            compressed_filename = f"{base_filename}_h264.{self.video_format}"
            compressed_path = os.path.join(videos_dir, compressed_filename)
            
            if self._compress_video_with_ffmpeg(output_path, compressed_path):
                # Check compressed file
                if os.path.exists(compressed_path) and os.path.getsize(compressed_path) > 0:
                    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)  # MB
                    compression_ratio = (1 - compressed_size / file_size) * 100
                    
                    print(f"\nCompression completed!")
                    print(f"Compressed output: {compressed_path}")
                    print(f"Compressed size: {compressed_size:.2f} MB")
                    print(f"Compression ratio: {compression_ratio:.1f}% reduction")
                    
                    # Optionally remove the original uncompressed file
                    if self.debug:
                        print(f"[VideoCreator] Keeping both original and compressed files")
                    
                    final_output_path = compressed_path
                else:
                    print("Warning: Compressed file was not created properly, using original file")
            else:
                print("Warning: Compression failed, using original file")
        
        return final_output_path
    
    def _get_fourcc(self) -> int:
        """Get the appropriate FourCC code for the video format."""
        if self.video_format == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v')
        elif self.video_format == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID')
        elif self.video_format == 'mov':
            return cv2.VideoWriter_fourcc(*'mp4v')
        else:
            # Default to mp4v
            return cv2.VideoWriter_fourcc(*'mp4v')
    
    def list_sessions(self) -> List[str]:
        """List all available session directories."""
        if not os.path.exists(self.base_dir):
            print(f"Base directory does not exist: {self.base_dir}")
            return []
        
        session_pattern = os.path.join(self.base_dir, "session_*")
        session_folders = glob.glob(session_pattern)
        session_folders.sort(key=os.path.getmtime, reverse=True)
        
        return session_folders


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Create videos from saved image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Create video from latest session at 10 FPS
            python create_video.py
            
            # Create video from specific session at 24 FPS
            python create_video.py --session /path/to/session_20231201_143022 --fps 24
            
            # Create video with compression
            python create_video.py --compress --compression-preset slow --compression-crf 23
            
            # Create video with custom settings and compression
            python create_video.py --fps 30 --format mp4 --quality 18 --compress --debug
            
            # List available sessions
            python create_video.py --list-sessions
                    """
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="output",
        help="Base directory containing session folders (default: output)"
    )
    
    parser.add_argument(
        "--session",
        type=str,
        help="Specific session directory to process (default: latest session)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output video (default: 10)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "avi", "mov"],
        help="Video format (default: mp4)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=23,
        help="Video quality - CRF value for H.264 (0-51, lower=better, default: 23)"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Apply additional ffmpeg compression with H.264 encoding"
    )
    
    parser.add_argument(
        "--compression-preset",
        type=str,
        default="slow",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="FFmpeg compression preset (default: slow)"
    )
    
    parser.add_argument(
        "--compression-crf",
        type=int,
        default=23,
        help="CRF value for ffmpeg compression (0-51, lower=better, default: 23)"
    )
    
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default="128k",
        help="Audio bitrate for compression (default: 128k)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output filename (default: auto-generated)"
    )
    
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available session directories and exit"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Create video creator instance
    creator = VideoCreator(
        base_dir=args.base_dir,
        fps=args.fps,
        video_format=args.format,
        video_quality=args.quality,
        compress=args.compress,
        compression_preset=args.compression_preset,
        compression_crf=args.compression_crf,
        audio_bitrate=args.audio_bitrate,
        debug=args.debug
    )
    
    # List sessions if requested
    if args.list_sessions:
        sessions = creator.list_sessions()
        if sessions:
            print(f"Available sessions in {args.base_dir}:")
            for i, session in enumerate(sessions, 1):
                session_name = os.path.basename(session)
                mod_time = datetime.fromtimestamp(os.path.getmtime(session))
                image_count = len(creator.get_image_files(session))
                print(f"  {i}. {session_name} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')}) - {image_count} images")
        else:
            print("No sessions found.")
        return
    
    # Create video
    try:
        output_path = creator.create_video(
            session_dir=args.session,
            output_filename=args.output
        )
        
        if output_path:
            print(f"\nSuccess! Video saved to: {output_path}")
        else:
            print("\nFailed to create video.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main() 