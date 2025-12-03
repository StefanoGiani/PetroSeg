# PetroSeg

Python app to label images of thin sections for segmentation.

## Installation

### 1. Clone the Repository

git https://github.com/StefanoGiani/PetroSeg.git
cd PetroSeg

### 2. Install Tkinter (if not already installed)

Tkinter is usually included with Python, but if it's missing, follow the instructions below:

🪟 Windows
pip install tk

🍎 macOS
If using Homebrew Python:
brew install python-tk
Or install Python from python.org which includes Tkinter.

🐧 Linux (Ubuntu/Debian)
sudo apt-get install python3-tk

### 3. Create a Virtual Environment (Recommended)
Creating a virtual environment helps isolate your app's dependencies.

🪟 Windows (CMD or PowerShell)
python -m venv venv
venv\Scripts\activate

🍎 macOS / 🐧 Linux
python3 -m venv venv
source venv/bin/activate

To deactivate the environment later, just run deactivate.

### 4. Install Python Dependencies

Make sure you have Python 3.8 or later installed. Then run:

pip install -r requirements.txt

### 5. Running the App

Once dependencies are installed, run the app with:
python PetroSeg.py


## Content

**assets/splash.png**
Image used for the splash screen of the application. It appears when the app starts and provides branding or visual context.

**PetroSeg.py**
The main entry point of the Python application. Contains the core logic and function calls to run the program.

**README.md**
This documentation file. It explains the purpose of the repository, how to install and use the software, and other relevant details.

**requirements.txt**
Lists all Python dependencies needed to run the project. Can be installed using pip install -r requirements.txt.

**License.txt**
File containing the details about the license.

**CITATION.cff**
Details on how to cite the software.

**PetroSeg_User_Manual.pdf**
Manual of the software.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licensest the code for non-commercial and academic purposes, provided that proper attribution is given. Commercial use is not permitted without prior written permission.

