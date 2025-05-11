# Dataset for Multi-modal Sensor

This repository contains the expected format and structure of datasets used for training models related to sensor signal processing, including temperature, pressure, and strain prediction.

> **Note:** The original datasets required for this project are not publicly uploaded.
---

## Dataset Overview

This project utilizes several Excel files for three main sensor components:

- Temperature Component
- Pressure Component
- Strain Component

---

## (1) Temperature Component

### File: `Temperature_data.xlsx`

| Column   | Description                      |
|----------|----------------------------------|
| `T(℃)`   | Temperature values (°C)          |
| `Z (Ω)`  | Impedance values (Ohms)          |

---

## (2) Pressure Component

### File: `Pressure_data.xlsx`

| Column           | Description                          |
|------------------|--------------------------------------|
| `T(℃)`           | Temperature values (°C)              |
| `Cp(F)`          | Capacitance values (Farads)          |
| `Pressure(kPa)`  | Pressure values (kilopascals)        |

---

## (3) Strain Component

### File 1: `Strain_data_without_strain.xlsx`

| Column   | Description                      |
|----------|----------------------------------|
| `T(℃)`   | Temperature values (°C)          |
| `Z (Ω)`  | Impedance values (Ohms)          |

### File 2: `Strain_data_with_strain.xlsx`

| Column   | Description                      |
|----------|----------------------------------|
| `T(℃)`   | Temperature values (°C)          |
| `Z (Ω)`  | Impedance values (Ohms)          |
| `Strain` | Strain values (unitless)         |

---

## Usage

Please ensure the above files are placed correctly before training or testing any model. This structure is required by the preprocessing and model input pipelines.
