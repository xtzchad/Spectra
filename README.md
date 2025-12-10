# 🎨 Spectra

**Intelligent image sorting by visual similarity**

Spectra automatically organizes your image collections by analyzing color, texture, and content similarity. Say goodbye to chaotic folders — let your images flow in visual harmony.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

## ✨ Features

- **🔍 Multi-dimensional visual analysis**: Extracts color histograms, spatial patterns, texture features, and brightness metrics
- **🧩 Smart clustering**: Uses DBSCAN algorithm to group visually similar images
- **🔄 Nearest-neighbor sorting**: Creates smooth visual transitions within and between clusters
- **🖥️ User-friendly GUI**: Clean Tkinter interface—no command line needed
- **🛡️ Safe operations**: Dry-run mode and automatic backups protect your files
- **📊 Detailed logging**: Real-time progress tracking and CSV mapping of all changes
- **⚙️ Customizable**: Adjustable similarity thresholds and filename prefixes

---

## 🚀 Quick start

### Option 1: Download pre-built binary (recommended). Currently only Windows build

**No Python installation required!**

1. Go to [Releases](https://github.com/xtzchad/Spectra/releases)
2. Download the latest version

### Option 2: Run from source

**Prerequisites:**
```bash
Python 3.8 or higher
```

1. **Clone the repository**
   ```
   git clone https://github.com/xtzchad/Spectra.git
   cd Spectra
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run Spectra**
   ```
   python Spectra.py
   ```

### Dependencies

```
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

---

## 📖 How it works

### 1. Feature extraction

Spectra analyzes each image across multiple dimensions:

- **RGB & HSV color histograms**: Captures overall color distribution
- **Spatial color features**: Tracks color placement in a 4×4 grid
- **Texture analysis**: Detects edges, patterns, and variance
- **Brightness & contrast**: Measures tonal characteristics

### 2. Intelligent clustering

Uses DBSCAN (Density-Based Spatial Clustering) to:
- Group images with similar visual characteristics
- Automatically determine optimal cluster count
- Handle outliers gracefully

### 3. Sequential ordering

- Sorts images within clusters using nearest-neighbor algorithm
- Orders clusters for smooth visual transitions
- Optimizes connections between cluster boundaries

### 4. Safe renaming

- Renames files sequentially (e.g., `001.jpg`, `002.jpg`, `003.jpg`)
- Creates backup of originals (optional)
- Generates CSV mapping file for reference

---

## 🎯 Use cases

- **📸 Photo collections**: Organize vacation photos by scene and color
- **🎨 Design assets**: Sort product images, textures, or color palettes
- **🖼️ Digital art**: Arrange artwork by style and composition
- **📱 Screenshots**: Group similar UI states or app screens
- **🏠 Home organization**: Sort scanned documents or family photos

---

## 🛠️ Usage guide

### Basic workflow

1. **Launch the application**
   ```bash
   python Spectra.py
   ```

2. **Select your image folder**
   - Click "Browse..." to choose your image directory
   - Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP

3. **Configure settings**
   - **File prefix**: Add a prefix to sorted filenames (optional)
   - **Similarity threshold**: Lower values = tighter grouping (0.005-0.05 typical)
   - **Auto-determine**: Let Spectra calculate optimal threshold
   - **Dry run**: Preview changes without modifying files
   - **Create backup**: Saves originals to `backup_originals/` folder

4. **Start sorting**
   - Click "Start Sorting"
   - Monitor progress in the log window
   - Review results and mapping file

### Advanced options

#### Custom similarity threshold

```python
# In code, adjust threshold parameter:
sorted_images = sort_with_tight_clustering(
    image_files, 
    similarity_threshold=0.015  # Lower = tighter clusters
)
```

#### Programmatic usage

```python
from Spectra import get_image_files, sort_with_tight_clustering, rename_images

# Load images
images = get_image_files("/path/to/images")

# Sort by similarity
sorted_images = sort_with_tight_clustering(images)

# Rename (with backup)
rename_images(sorted_images, prefix="sorted_", backup=True)
```

---

## 📊 Output files

After processing, Spectra generates:

```
your-image-folder/
├── sorted_001.jpg          # Renamed images in order
├── sorted_002.jpg
├── sorted_003.jpg
├── ...
├── rename_mapping.csv      # Original → New name mapping
└── backup_originals/       # Original files (if backup enabled)
    ├── original_name1.jpg
    ├── original_name2.jpg
    └── ...
```

### Mapping file format

```csv
Original Name,New Name
"IMG_5234.jpg","sorted_001.jpg"
"DSC_0891.jpg","sorted_002.jpg"
"photo.png","sorted_003.png"
```

---

## 🎛️ Configuration

### Similarity threshold guidelines

| Threshold | Clustering behavior |
|-----------|---------------------|
| 0.005-0.01 | Very tight—only nearly identical images grouped |
| 0.01-0.02 | Moderate—similar colors and compositions |
| 0.02-0.05 | Loose—broader visual themes |
| Auto | Spectra calculates based on your dataset |

### Performance notes

- **Processing time**: ~0.5-1 second per image (depends on resolution)
- **Memory usage**: ~50-100MB per 1000 images
- **Optimal batch size**: Up to 5000 images per folder

---

## 🔒 Safety features

- ✅ **Dry run mode**: Preview all changes before committing
- ✅ **Automatic backups**: Original files preserved in separate folder
- ✅ **Mapping file**: CSV log of all filename changes
- ✅ **Non-destructive**: Images are renamed, never modified
- ✅ **Error handling**: Skips problematic images with warnings

---

## 🔧 Building from source

### Creating standalone executables

Want to compile your own binary? Here's how:

#### Prerequisites

```bash
pip install pyinstaller
```

#### Windows

```bash
pyinstaller --onefile --windowed --name Spectra --icon=icon.ico Spectra.py
```

The executable will be in `dist/Spectra.exe`

#### macOS

```bash
pyinstaller --onefile --windowed --name Spectra --icon=icon.icns Spectra.py
```

The app bundle will be in `dist/Spectra.app`

#### Linux

```bash
pyinstaller --onefile --name Spectra Spectra.py
```

The executable will be in `dist/Spectra`

#### Advanced build options

For a smaller executable with optimizations:

```bash
pyinstaller --onefile \
            --windowed \
            --name Spectra \
            --strip \
            --exclude-module matplotlib \
            --exclude-module pytest \
            Spectra.py
```

**Build options explained:**
- `--onefile`: Bundle everything into a single executable
- `--windowed`: Hide console window (GUI only)
- `--name`: Set output filename
- `--icon`: Add custom icon (`.ico` for Windows, `.icns` for macOS)
- `--strip`: Remove debug symbols (smaller size)
- `--exclude-module`: Exclude unnecessary packages

#### Troubleshooting builds

**"Failed to execute script" error:**
```bash
# Build without --windowed to see error messages
pyinstaller --onefile --name Spectra Spectra.py
```

**Missing modules in compiled version:**
```bash
# Add hidden imports
pyinstaller --onefile --hidden-import=PIL._tkinter_finder Spectra.py
```

**Antivirus false positives:**
- This is common with PyInstaller executables
- Sign your executable with a code signing certificate
- Or add your build folder to antivirus exclusions during development

---

## 🤝 Contributing

Contributions are welcome!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Pillow](https://python-pillow.org/) for image processing
- Powered by [scikit-learn](https://scikit-learn.org/) for clustering algorithms
- UI created with [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

## 📧 Contact

**Questions? Suggestions? Issues?**

- 🐛 [Report a bug](https://github.com/xtzchad/Spectra/issues)
- 💡 [Request a feature](https://github.com/xtzchad/Spectra/issues)
- 📖 [Documentation](https://github.com/xtzchad/Spectra/wiki)

---

<p align="center">
  Made with ❤️ for photographers, designers, and digital hoarders everywhere
</p>

<p align="center">
  <sub>Star ⭐ this repo if you find it useful!</sub>
</p>
