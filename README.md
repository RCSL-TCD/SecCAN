# Secure CAN Controller with Integrated IDS  

This repository contains the hardware implementation of a **Secure CAN Controller** with an **Integrated Intrusion Detection System (IDS)**. The design features the ability to flag suspicious messages for the receiving ECU, enhancing the security of the Controller Area Network (CAN).  

## Key Features  
- **Extension of Open-Source CAN Controller**:  
  Built upon an existing Verilog-based open-source CAN controller implementation.  
- **Integrated IDS**:  
  - The IDS is embedded within the **BSP module** of the controller.  
  - It appends a flag bit to each message for real-time attack detection.  
- **Advanced Threat Detection**:  
  - A 4-bit quantized Multi-Layer Perceptron (MLP) serves as the IDS, operating as a binary classifier.  
  - Trained to detect **DoS/Flooding**, **Fuzzing**, and **Malfunction** attacks.  
  - Utilizes datasets like the **Open Car Hacking Dataset** and **Survival Analysis Dataset** for robust training.  

## Technical Highlights  
- **Hardware IP for Quantized IDS**:  
  Developed using the open-source **FINN toolchain** from AMD Research Labs.  
- **Training with Quantization-Aware Tools**:  
  The IDS model is trained using **Brevitas**, an open-source quantization-aware training library from AMD Research Labs.  

This project demonstrates a novel approach to enhancing the security of CAN systems by integrating machine learning-based intrusion detection into the hardware.  

