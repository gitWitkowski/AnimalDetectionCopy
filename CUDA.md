# **How to Check if CUDA Can Be Installed on Your System**


## **Step 1: Check Your GPU Model**
1. Open a terminal (Linux/macOS) or Command Prompt (Windows).
2. Run the following command to list your GPU details:
   ```sh
   nvidia-smi
   ```
3. Look for your **GPU name** in the output.
4. Check if your GPU is CUDA-compatible by visiting the official list of CUDA-supported GPUs:  
   🔗 [CUDA-Enabled GPUs](https://developer.nvidia.com/cuda-gpus)

   **If your GPU is listed, it supports CUDA. Otherwise, you will **

---

## **Step 2: Check if NVIDIA Drivers Are Installed**
1. Run the following command:
   ```sh
   nvidia-smi
   ```
2. If the command works and shows GPU details, your NVIDIA drivers are installed.  
   If not, install the latest drivers from:  
   🔗 [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)

---

## **Step 3: Check Operating System Compatibility**
CUDA supports the following operating systems:
- **Windows:** Windows 10 or later (64-bit)
- **Linux:** Ubuntu, CentOS, RHEL, Fedora, Debian, etc.
- **macOS:** CUDA is **not** supported on macOS (NVIDIA no longer provides drivers for recent macOS versions).

Check the official CUDA documentation for supported OS versions:  
🔗 [CUDA Installation Guide](https://docs.nvidia.com/cuda/index.html)

---

## **Step 4: Check If Your System Has a Compatible Compiler**
CUDA requires a compatible C++ compiler:
- **Windows:** Microsoft Visual Studio (VS) is required.
- **Linux/macOS:** GCC (GNU Compiler Collection) is needed.

To check your compiler version:
```sh
gcc --version   # For Linux/macOS
cl              # For Windows (in Visual Studio Developer Command Prompt)
```
Compare your compiler version with the **NVIDIA CUDA Toolkit Compatibility Chart**:  
🔗 [CUDA Compiler Compatibility](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

---

## **Step 5: Check Available Disk Space**
Ensure you have at least **3-5GB** of free disk space for CUDA installation.

- **Windows:** Check using **File Explorer** → Right-click on drive → "Properties".
- **Linux/macOS:** Run:
  ```sh
  df -h
  ```

---

## **Step 6: Check If Your System Has CUDA Installed (Optional)**
If CUDA is already installed, check its version:
```sh
nvcc --version
```
or
```sh
nvidia-smi
```

---

## **Conclusion**
If your GPU is CUDA-compatible, drivers are installed, OS and compiler meet requirements, and you have enough disk space, **you can install CUDA**. 🎉

To install, download the CUDA Toolkit from:  
🔗 [NVIDIA CUDA Toolkit Download](https://developer.nvidia.com/cuda-12-5-0-download-archive)  

Although there are higher versions, the recommended version is CUDA 12.5.
Although there are higher versions, the recommended version is CUDA 12.5.
