# 🌍 Physics-Informed Neural Network for Temperature Modeling in Europe

This project explores the use of **Physics-Informed Neural Networks (PINNs)** to model the spatiotemporal evolution of surface temperature across Europe, using real observational data from the **ERA5 climate reanalysis**.

---

## 📌 Summary

- Uses the **2D heat equation** as a physical constraint:
  \[
  \frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
  \]
- Trained on real-world ERA5 temperature data (2m temperature, 6-hourly over 5 days)
- Combines data fitting loss with PDE residual loss using **autograd in PyTorch**
- Implements learnable physical parameters:
  - **α** (thermal diffusivity)
  - **λ** (PDE constraint weight)
- Supports **fully connected neural networks (FCNN)** architecture

---

## 📡 Data

- Source: [ERA5 Reanalysis](https://cds.climate.copernicus.eu/)
- Variable: 2-meter air temperature
- Temporal resolution: 6-hourly
- Spatial coverage: Europe (35°N–60°N, -10°E–30°E)

---

## 🔧 Techniques Used

- PyTorch for custom PINN training
- Autograd for computing first and second derivatives w.r.t. input coordinates
- Joint optimization of physical and data-driven objectives
- Learnable `log(α)` and `log(λ)` to ensure positivity and dynamic balancing

---

## 📉 Results

- Model successfully learned to **approximate spatial temperature structure**
- PDE residuals showed **partial enforcement** of physical consistency
- Final predictions showed **some realistic patterns**, but lacked full smoothness and accuracy
- The model struggled with sharp gradients and temporal generalization

![PINN Prediction](output_pred.png)
*PINN-predicted temperature field (Day 3)*

![Residuals](output_residual.png)
*PDE residual magnitude — large values indicate weak constraint satisfaction*

---

## 💡 Reflections

While the results were mixed, the project provided valuable insights into:

- Training dynamics of PINNs on real (non-synthetic) data
- Numerical challenges of enforcing PDE constraints in high dimensions
- Trade-offs between model flexibility and physical fidelity

