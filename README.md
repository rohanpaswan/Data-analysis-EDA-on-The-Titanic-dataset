End-to-End Exploratory Data Analysis (EDA) on the Titanic Dataset
Project Objective: To perform a comprehensive, step-by-step exploratory data analysis to understand the key factors that influenced survival on the Titanic. This notebook will serve as a complete guide, covering data loading, cleaning, analysis, feature engineering, and visualization, with theoretical explanations at each stage.
Theoretical Concept: What is Exploratory Data Analysis (EDA)?
Exploratory Data Analysis is the crucial process of performing initial investigations on data to discover patterns, spot anomalies, test hypotheses, and check assumptions with the help of summary statistics and graphical representations. It is not about formal modeling or hypothesis testing; rather, it is about getting to know your data before you start building models.

Why is it important?

1 Understand the Data: It helps you understand the variables and their relationships.

2 Data Cleaning: It reveals missing values, outliers, and other inconsistencies that need to be handled.

3 Feature Selection: It helps identify which variables are the most important for your problem (feature engineering and selection)
.
4 Assumption Checking: It allows you to check assumptions that are required for certain machine learning models (e.g., normality, linearity).
Libraries Used: Pandas and Seaborn
Pandas: This is a powerful Python library for data manipulation and analysis. It provides data structures like DataFrames, which are essential for working with tabular data. We used Pandas to load the dataset, handle missing values, and perform various data transformations.


Seaborn: Built on top of Matplotlib, Seaborn is a statistical data visualization library. It provides a high-level interface for drawing attractive and informative statistical graphics. We used Seaborn to create various plots like countplots, histograms, boxplots, and barplots to visualize the distributions and relationships within the data.

The dataset contains 891 entries (passengers) and 12 columns.
Missing Values Identified: Age, Cabin, and Embarked have missing values. Cabin is missing a significant amount of data (~77%), which will require special attention.

Survived: About 38.4% of passengers in this dataset survived.
Age: The age ranges from ~5 months to 80 years, with an average age of about 30.
Fare: The fare is highly skewed, with a mean of  32butamedianofonly 14.45. The maximum fare is over $512, indicating the presence of extreme outliers.
Data Cleaning
Before analysis, we must handle the missing values we identified.
Theoretical Concept: Missing Value Imputation
Imputation is the process of replacing missing data with substituted values. The strategy depends on the data type and its distribution:

1 Numerical Data: For skewed distributions (like Age and Fare), using the median is more robust than the mean because it is not affected by outliers.
2 Categorical Data: A common strategy is to fill with the mode (the most frequent value).
3 High Cardinality/Too Many Missing Values: For columns like Cabin, where most data is missing, imputing might not be effective. We could either drop the column or engineer a new feature from it (e.g., Has_Cabin).

Univariate Analysis

We analyze each variable individually to understand its distribution.
#### **Theoretical Concept: Univariate Analysis**

This is the simplest form of data analysis, where the data being analyzed contains only one variable. The main purpose is to describe the data and find patterns within it.
- **For Categorical Variables:** We use frequency tables, bar charts (`countplot`), or pie charts to see the count or proportion of each category.
- **For Numerical Variables:** We use histograms (`histplot`) or kernel density plots (`kdeplot`) to understand the distribution, and box plots (`boxplot`) to identify the central tendency, spread, and outliers.
print("Analyzing categorical features:")


**Key Insights (Categorical):**
- **Survival:** Most passengers (over 500) did not survive.
- **Pclass:** The 3rd class was the most populated, followed by 1st and then 2nd.
- **Sex:** There were significantly more males than females.
- **Embarked:** The vast majority of passengers embarked from Southampton ('S').
- **SibSp & Parch:** Most passengers traveled alone.


# Plotting Age distribution
# Plotting Fare distribution

**Key Insights (Numerical):**
- **Age:** The distribution peaks around the 20-30 age range. Remember we filled missing values with the median (28), which contributes to the height of that central bar.
- **Fare:** The distribution is heavily right-skewed, confirming that most tickets were cheap, with a few very expensive exceptions.
**Key Insights (Numerical):**
- **Age:** The distribution peaks around the 20-30 age range. Remember we filled missing values with the median (28), which contributes to the height of that central bar.
- **Fare:** The distribution is heavily right-skewed, confirming that most tickets were cheap, with a few very expensive exceptions.
Key Insights (Numerical):

1 Age: The distribution peaks around the 20-30 age range. Remember we filled missing values with the median (28), which contributes to the height of that central bar.
2 Fare: The distribution is heavily right-skewed, confirming that most tickets were cheap, with a few very expensive exceptions.
Theoretical Concept: Bivariate Analysis
This type of analysis involves two different variables, and its main purpose is to find relationships between them.

1 Categorical vs. Numerical: To compare a numerical variable across different categories, we often use bar plots (barplot) that show the mean (or another estimator) of the numerical variable for each category. We can also use box plots or violin plots.
2 Categorical vs. Categorical: We can use stacked bar charts or contingency tables (crosstabs).
Numerical vs. Numerical: A scatter plot is the standard choice, with a correlation matrix being used to quantify the relationship.
Deeper Dive: Outlier Analysis for 'Fare'
The .describe() function and histogram showed that Fare has extreme outliers. Let's visualize this clearly with a box plot.


Observation: The box plot confirms the presence of significant outliers. Most fares are concentrated below $100, but there are several fares extending far beyond, with some even exceeding $500. These are likely first-class passengers who booked luxurious suites. For some machine learning models, handling these outliers (e.g., through log transformation) would be an important step.
 Feature Engineering
Now, we'll create new features from the existing ones to potentially uncover deeper insights and provide more useful information for a machine learning model.
Theoretical Concept: Feature Engineering
Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. A good feature should be relevant to the problem and easy for a model to understand.

Common Techniques:

1 Combining Features: Creating a new feature by combining others (e.g., SibSp + Parch = FamilySize).
2 Extracting from Text: Pulling out specific information from a text feature (e.g., extracting titles from the Name column).
3 Binning: Converting a continuous numerical feature into a categorical one (e.g., binning Age into groups like 'Child', 'Adult', 'Senior').
## 1. Create a 'FamilySize' Column

## 2. Create an 'Isalone' Feature

# Analyze the new family-related features against survival

# Survival Rate by FamilySize


# Survival Rate by IsAlone

**Insight:**
- Passengers who were alone (`IsAlone=1`) had a lower survival rate (~30%) than those in small families.
- Small families of 2 to 4 members had the highest survival rates.
- Very large families (5 or more) had a very poor survival rate. This might be because it was harder for large families to stay together and evacuate.

# 3. Extract 'Title' from the 'Name' column
Matches a space.
* Titles in the names are usually preceded by a space. ([A-Za-z]+): This is the capturing group.
* [A-Za-z]+: Matches one or more uppercase or lowercase letters. This captures the title itself (like Mr, Mrs, Miss, etc.).
* .: Matches a literal dot (.) which usually follows the title.

# Simplify the titles by grouping rare ones into a 'Rare' category


# Let's see the survival rate by the new, cleaned titles

Insight: The Title feature gives us powerful information. 'Mrs' and 'Miss' (females) had high survival rates. 'Mr' (males) had a very low survival rate. 'Master' (young boys) had a significantly higher survival rate than 'Mr', reinforcing the 'children first' idea. The 'Rare' titles, often associated with nobility or status, also had a mixed but generally higher survival rate than common men.
 Multivariate Analysis
# Survival rate by Pclass and Sex

# Insights: Females in all classes had a significantly higher survival rate than males.
# Violin plot to see age distribution by sex and survival statu 

 Insight from Violin Plot:

* For males, the peak of the distribution for survivors (orange) is at a very young age (children), while the peak for non-survivors is in the 20-30 range.
* For females, the distribution of survivors is much broader, indicating that females of most ages had a good chance of surviving.

   Correlation Analysis

Interpretation of the Heatmap:

* Survived has a notable positive correlation with Fare and Has_Cabin, and a negative correlation with Pclass and our new IsAlone feature.
* Pclass and Fare are strongly negatively correlated, which makes sense (1st class = high fare).
* Our new FamilySize feature is composed of SibSp and Parch, so it's highly correlated with them by definition.
Step 9: Final Conclusion and Summary of Insights
This end-to-end EDA has provided a deep understanding of the Titanic dataset. Our analysis confirms the "women and children first" narrative and highlights the stark social inequalities of the time. Through feature engineering, we've created even more powerful predictors for a potential machine learning model.

Key Findings:

1 Strongest Predictors of Survival:

* Title & Sex: Being female ('Mrs', 'Miss') was the single most significant advantage. Our engineered Title feature captures this nuance better than Sex alone, also showing that young boys ('Master') had a much higher survival rate than adult men ('Mr').
* Passenger Class: There was a clear survival hierarchy: 1st > 2nd > 3rd class.
* Age: Children and infants had a higher survival rate.

2 Other Influential Factors:

* Family Size: Traveling in a small family (2-4 members) increased survival chances, while traveling alone or in a very large family decreased them.
* Fare/Cabin: Having a cabin (and thus paying a higher fare) was strongly correlated with survival, acting as a proxy for wealth and passenger class.
* Port of Embarkation: Passengers from Cherbourg ('C') had a higher survival rate, possibly because a higher proportion of them were in 1st class.


These insights are fundamental for the next step in the data science pipeline: building a predictive machine learning model to forecast survival.
y-profiling:
* y-profiling automatically generates an in-depth data analysis report
* Summarizes data types, missing values, and distributions
* Shows correlations and potential data quality issues
* Helps understand the structure and characteristics of a dataset at a glance
from ydata_profiling import ProfileReport


