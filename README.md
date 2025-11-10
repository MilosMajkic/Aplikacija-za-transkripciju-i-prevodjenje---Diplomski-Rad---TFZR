# Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR

##  Overview
Aplikacija za transkripciju i prevodjenje je diplomski rad koji koristi tehnologiju za transkripciju govora i prevodjenje teksta. Ova aplikacija je napravljena za razne svrhe, uključujući akademska istraživanja, profesionalne preglede i lične projekte. Aplikacija koristi modernu tehnologiju za vizuelizaciju zvuka i analizu zvuka, čime se olakšava proces transkripcije i prevodjenja.

##  Features
-  Transkripcija govora u tekst
-  Prevodjenje teksta u više jezika
-  Vizuelizacija zvuka
-  Analiza zvuka
-  Generiranje SRT datoteka

##  Tech Stack
- **Programming Language:** Python
- **Frameworks & Libraries:**
  - Streamlit
  - Pandas
  - Altair
  - Faster Whisper
  - NumPy
  - Matplotlib
  - Librosa
- **System Requirements:**
  - Python 3.8+
  - Streamlit
  - Faster Whisper
  - NumPy
  - Matplotlib
  - Librosa

##  Installation

### Prerequisites
- Python 3.8+
- pip

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR.git

# Navigate to the project directory
cd Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR

# Install the required packages
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
```

### Alternative Installation Methods
- **Docker:** (if applicable)
  ```bash
  docker build -t transcribe-app .
  docker run -p 8501:8501 transcribe-app
  ```

##  Usage

### Basic Usage
```python
# Example usage of the application
import streamlit as st
import pandas as pd
import altair as alt
from faster_whisper import WhisperModel

# Load the model
model = WhisperModel("small")

# Transcribe audio
segments, _ = model.transcribe("path/to/audio/file.wav")

# Print the transcription
for segment in segments:
    print(segment.text)
```

### Advanced Usage
- **Configuration Options:**
  - Set environment variables for model and compute type:
    ```bash
    export WHISPER_MODEL="small"
    export WHISPER_COMPUTE="int8"
    ```

- **API Documentation:**
  - (if applicable)

##  Project Structure
```
Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR/
│
├── app.py
├── pokretanje.txt
├── .idea/
│   └── .gitignore
└── requirements.txt
```

##  Configuration
- **Environment Variables:**
  - `WHISPER_MODEL`: Default model size (tiny, base, small, medium, large-v3)
  - `WHISPER_COMPUTE`: Compute type (int8, float16, int8_float16)

- **Configuration Files:**
  - (if applicable)

##  Contributing
- Fork the repository
- Create a new branch (`git checkout -b feature/your-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin feature/your-feature`)
- Open a Pull Request

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Authors & Contributors
- **Maintainers:** [Your Name]
- **Contributors:** [List of contributors]

##  Issues & Support
- Report issues on the [GitHub Issues page](https://github.com/yourusername/Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR/issues)
- Get help on the [GitHub Discussions page](https://github.com/yourusername/Aplikacija-za-transkripciju-i-prevodjenje---Diplomski-Rad---TFZR/discussions)

##  Roadmap
- **Planned Features:**
  - Integration with more translation APIs
  - Improved audio visualization
  - Support for more audio formats

- **Known Issues:**
  - (if applicable)

- **Future Improvements:**
  - Enhanced user interface
  - Better performance optimizations

---

**Additional Guidelines:**
- Use modern markdown features (badges, collapsible sections, etc.)
- Include practical, working code examples
- Make it visually appealing with appropriate emojis
- Ensure all code snippets are syntactically correct for Python
- Include relevant badges (build status, version, license, etc.)
- Make installation instructions copy-pasteable
- Focus on clarity and developer experience
