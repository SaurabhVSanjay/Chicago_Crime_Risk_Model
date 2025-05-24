# Chicago_Crime_Risk_Model
A PySpark-based machine learning pipeline to classify violent and non-violent crimes in Chicago using the official crime dataset. Includes full ETL, EDA, feature engineering, model building, and visual evaluation.

# Chicago Crime Risk Model

This project builds a scalable machine learning pipeline using PySpark to classify violent vs non-violent crimes reported in Chicago since 2001. It includes end-to-end data engineering, exploratory data analysis, feature engineering, model building, ensemble voting, evaluation, and optimization.

---

## Project Summary

We analyze 300K+ historical records from the Chicago Police Department to uncover spatial, temporal, and behavioral patterns in crime. Our objective is to build a risk model to support data-driven decision-making for policy makers and law enforcement.

**Key Outcomes:**
- Gradient Boosted Tree classifier with 76% accuracy
- Ensemble model improves violent crime precision to 79%
- Identifies theft-heavy and vice-heavy hotspots for focused intervention
- Reveals spatial clustering of violent crimes

---

## Dataset

- **Source**: [Chicago Data Portal - Crimes 2001 to Present](https://data.cityofchicago.org/)
- **Records**: ~412,000 crime incidents
- **Features**:
  - Temporal: `date`, `hour`, `weekday`, `month`, `year`
  - Spatial: `latitude`, `longitude`, `ward`, `district`, `community_area`
  - Behavioral: `primary_type`, `location_description`, `arrest`, `domestic`

---

## Technologies Used

- PySpark
- Apache Spark 3.5 (via Databricks)
- Matplotlib / Seaborn (for visualization)
- XGBoost
- Pandas & NumPy (for intermediate analysis)
- Machine Learning: GBT, Random Forest, Logistic Regression, Ensemble Voting

---

## ML Model Pipeline

- Cleaned and indexed critical fields
- Engineered new features: hour, weekday, year, etc.
- Binary classification:
  - `1`: Violent crime (e.g., homicide, robbery, assault)
  - `0`: Non-violent crime
- Models:
  - **Gradient Boosted Tree (GBT)** - Base model
  - **Voting Ensemble (LR + RF + XGBoost)** - Final model
- Evaluation:
  - Confusion Matrix
  - Precision-Recall Curve
  - Class-wise F1 scores

---

## Results Snapshot

| Metric        | GBT        | Ensemble Voting |
|---------------|------------|-----------------|
| Accuracy      | 76.2%      | 75.8%           |
| Violent Recall| 54%        | 55%             |
| Violent Precision| 78%     | 79%             |
| Non-Violent F1| 0.82       | 0.81            |

---

## 📁 Folder Structure

