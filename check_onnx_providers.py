import onnxruntime as ort
import os

def check_providers():
    available = ort.get_available_providers()
    print(f"Available providers: {available}")
    
    functional = []
    # Create a dummy model or just try to init session with each
    # Actually, just trying to init an InferenceSession with a provider is the best test.
    # But we need a dummy model. 
    # Let's just try to see if we can catch the error logs.
    
    for p in available:
        if p == 'CPUExecutionProvider':
            functional.append(p)
            continue
        try:
            # Try to create a session with an empty model or something
            # To avoid needing a real model file, we can just check if the DLLs load
            # ONNX Runtime doesn't have a simple 'is_provider_functional' API.
            # But we can try to create a session with a non-existent file just to see if the provider is accepted.
            # Actually, the error happens during DLL loading which is part of provider init.
            ort.InferenceSession(b"dummy", providers=[p])
        except Exception as e:
            if "FAIL" in str(e) or "Error loading" in str(e):
                print(f"Provider {p} is broken.")
            else:
                # If it's just 'file not found' or similar, the provider might be fine
                print(f"Provider {p} seems to be loadable (received: {type(e).__name__})")
                functional.append(p)
    
    print(f"Functional providers: {functional}")

if __name__ == "__main__":
    check_providers()
