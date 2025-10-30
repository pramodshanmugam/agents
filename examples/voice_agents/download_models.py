#!/usr/bin/env python3
"""
Manual script to download required models for LiveKit agents
"""
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

def download_turn_detector_models():
    """Download the turn detector models"""
    print("Downloading turn detector models...")
    
    try:
        # Download the ONNX model
        onnx_path = hf_hub_download(
            repo_id="livekit/turn-detector",
            filename="model.onnx",
            subfolder="onnx",
            revision="v0.3.0-intl",
            local_files_only=False  # Allow online download
        )
        print(f"✓ Downloaded ONNX model to: {onnx_path}")
        
        # Download the tokenizer
        tokenizer_path = AutoTokenizer.from_pretrained(
            "livekit/turn-detector",
            revision="v0.3.0-intl",
            local_files_only=False  # Allow online download
        )
        print("✓ Downloaded tokenizer")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading turn detector models: {e}")
        return False

def download_silero_vad():
    """Download Silero VAD model"""
    print("Downloading Silero VAD model...")
    
    try:
        # The Silero VAD model should be downloaded automatically by the silero plugin
        # Let's try to trigger it manually
        from livekit.plugins import silero
        
        # This should download the model if it doesn't exist
        vad = silero.VAD.load()
        print("✓ Silero VAD model loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error with Silero VAD: {e}")
        return False

if __name__ == "__main__":
    print("Starting model download...")
    
    success1 = download_turn_detector_models()
    success2 = download_silero_vad()
    
    if success1 and success2:
        print("\n🎉 All models downloaded successfully!")
        print("You can now run: python3 basic_agent.py console")
    else:
        print("\n❌ Some models failed to download. Check the errors above.")