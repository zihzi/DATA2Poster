import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load the dataset
df = pd.read_csv("data/movies-small.csv")

# Set style
sns.set(style="whitegrid")

# 1. Budget vs Gross Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Production Budget", y="Worldwide Gross", hue="Genre", alpha=0.7)
plt.title("Production Budget vs. Worldwide Gross by Genre")
plt.tight_layout()
plt.savefig("chart1_budget_vs_gross.png")
plt.close()

# 2. IMDB Rating by Genre
plt.figure(figsize=(6, 4))
avg_imdb = df.groupby("Genre")["IMDB Rating"].mean().sort_values()
sns.barplot(x=avg_imdb.values, y=avg_imdb.index, palette="viridis")
plt.title("Average IMDB Rating by Genre")
plt.xlabel("Average IMDB Rating")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig("chart2_avg_imdb_by_genre.png")
plt.close()

# 3. Rotten Tomatoes Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["Rotten Tomatoes Rating"], bins=20, kde=True, color="salmon")
plt.title("Distribution of Rotten Tomatoes Ratings")
plt.xlabel("Rotten Tomatoes Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart3_rotten_rating_distribution.png")
plt.close()

# Generate PDF Poster
pdf = FPDF(orientation='P', unit='mm', format='A3')
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Title
pdf.set_font("Arial", 'B', 24)
pdf.cell(0, 15, "Exploratory Data Analysis of Movie Dataset", ln=True, align='C')

# Improved Introduction
intro = (
    "This poster presents the exploratory data analysis of a dataset containing 274 movies, "
    "covering information such as production budgets, worldwide gross revenue, release years, content ratings, "
    "genres, and both critic (Rotten Tomatoes) and audience (IMDB) scores.\n\n"
    "Our analysis focuses on identifying relationships between budget and gross revenue, how genres influence audience ratings, "
    "and the overall distribution of critical reception.\n\n"
    "We employed scatter plots, bar charts, and histograms to reveal patterns such as:\n"
    "- Some low-budget films perform exceptionally well at the global box office.\n"
    "- Genres like Drama and Thriller tend to receive higher IMDB ratings.\n"
    "- Most films in the dataset are positively rated by Rotten Tomatoes, though a few outliers exist.\n\n"
    "These insights can be useful for filmmakers, analysts, and marketers interested in understanding factors tied to movie success."
)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, intro)
pdf.ln(5)

# Chart 1
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Chart 1: Production Budget vs. Worldwide Gross", ln=True)
pdf.image("chart1_budget_vs_gross.png", w=250)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, 
    "This scatter plot shows that while higher budgets often lead to higher gross revenue, there are exceptions. "
    "Some low-budget films also perform very well. Genre plays a noticeable role in this variability.")

# Chart 2
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Chart 2: Average IMDB Rating by Genre", ln=True)
pdf.image("chart2_avg_imdb_by_genre.png", w=250)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, 
    "This bar chart shows that genres like Drama and Thriller have higher average ratings, "
    "indicating strong audience appreciation for these types of stories.")

# Chart 3
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Chart 3: Distribution of Rotten Tomatoes Ratings", ln=True)
pdf.image("chart3_rotten_rating_distribution.png", w=250)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, 
    "The histogram reveals a positive skew in Rotten Tomatoes ratings, with many movies receiving high scores, "
    "although a few perform poorly.")

# Conclusion
pdf.ln(5)
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Conclusion", ln=True)
pdf.set_font("Arial", size=12)
conclusion = (
    "From this analysis, we conclude that production budget does not guarantee high revenue. "
    "Audience ratings differ across genres, and most movies tend to have positive critical reception. "
    "These insights could help guide producers and marketers in decision-making."
)
pdf.multi_cell(0, 10, conclusion)

# Save PDF
pdf.output("movie_eda_poster.pdf")
print("✅ Poster saved as 'movie_eda_poster.pdf'")
