"""Example usage of the Iris TTS model."""

from src.iris.model import TTSPipeline

def main():
    """Example TTS pipeline usage."""
    # Initialize the TTS pipeline
    pipeline = TTSPipeline()
    
    # Compile models
    pipeline.compile()
    
    # Print model summaries
    pipeline.summary()
    
    # Example synthesis (will raise NotImplementedError until implemented)
    # audio = pipeline.synthesize("Hello, world!")
    # print(f"Generated audio shape: {audio.shape}")

if __name__ == "__main__":
    main()

