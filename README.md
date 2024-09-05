# Instagram-Reach-Analysis-using-Python
Welcome to the **Instagram Reach Analysis** repository! This project aims to help content creators and data enthusiasts understand how Instagram's reach evolves over time, providing insights into post performance, impressions, and the factors influencing them.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Predicting Instagram Reach](#predicting-instagram-reach)
- [Technologies Used](#technologies-used)
- [License](#license)

## Overview

Instagram is one of the most popular social media platforms today, used by millions of people to promote their businesses, build portfolios, blog, and create content. As Instagram’s algorithms constantly evolve, the reach of posts fluctuates, affecting content visibility. This project focuses on understanding these changes through data analysis and predicting post reach.

This project includes:
- Data analysis of Instagram metrics such as impressions from home, hashtags, explore, and other sources.
- Visualization of key insights using Python libraries such as Matplotlib, Seaborn, and Plotly.
- WordClouds to understand caption and hashtag usage trends.
- Predictive modeling using machine learning to forecast post reach.

## Dataset

The dataset contains manually collected Instagram post metrics, including:
- **Impressions** (From Home, Hashtags, Explore, etc.)
- **Engagement** (Likes, Comments, Shares, Saves, Follows)
- **Post details** (Captions, Hashtags)

You can download the dataset from [here](https://statso.io/instagram-reach-analysis-case-study/).

## Installation

Clone this repository and install the necessary Python dependencies.

```bash
git clone https://github.com/Omkar1279/instagram-reach-analysis.git
cd instagram-reach-analysis
```

### Dependencies
The main libraries required for this project are:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- sklearn
- wordcloud

## Usage

1. Clone the repository.
2. Download the dataset or use your Instagram data.
3. Open the Jupyter notebook or run the Python script for analysis.

### Running the analysis

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
data = pd.read_csv('Instagram.csv', encoding='latin1')

# Clean the data and perform analysis
data = data.dropna()
```

## Visualizations

This project generates several visualizations to help you understand Instagram reach trends:
- **Distribution of Impressions**: Understand the reach from home, hashtags, explore, etc.
- **Donut Plot of Impressions by Source**: Visualize which sources contribute the most to post reach.
- **WordCloud of Captions and Hashtags**: Identify frequently used words in captions and hashtags.
- **Scatter Plots**: Understand relationships between likes, impressions, and engagement.

### Example: Distribution of Impressions

```python
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()
```

## Predicting Instagram Reach

Using **PassiveAggressiveRegressor** from Scikit-learn, the project predicts the reach of Instagram posts based on features such as:
- Impressions
- Saves, Likes, Shares, Comments
- Profile Visits, Follows

Example code snippet for model training:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Train-test split
X = data[['Impressions', 'Saves', 'Likes', 'Shares', 'Comments']]
y = data['Follows']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = PassiveAggressiveRegressor()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
```

## Technologies Used

- **Python**: Core language used for data analysis and modeling.
- **pandas, numpy**: For data manipulation and analysis.
- **matplotlib, seaborn, plotly**: Data visualization libraries.
- **sklearn**: For machine learning and predictions.
- **wordcloud**: For text analysis of captions and hashtags.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/yourusername/instagram-reach-analysis/issues) if you want to contribute.

Happy analyzing! 🎉
