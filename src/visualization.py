import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_percentage(df):
    missing_percent = (
        df.isnull().sum() / len(df)
    ) * 100
    missing_percent = missing_percent[missing_percent > 0]
    plt.figure(figsize=(10,5))
    missing_percent.plot(
        kind='bar',
        color='blue'
    )
    plt.title("Missing Value Percentage")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_genre_distribution(df):
    plt.figure(figsize=(12,6))
    df['listed_in'].value_counts().head(15).plot(kind='bar')
    plt.title("Top 15 Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_type_distribution(df):
    plt.figure(figsize=(6,4))
    df['type'].value_counts().plot(kind='bar')
    plt.title("Movie vs TV Show Distribution")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_release_year_distribution(df):
    plt.figure(figsize=(12,6))
    df['release_year'].hist(bins=25)
    plt.title("Release Year Distribution")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(df):
    plt.figure(figsize=(10,6))
    df['rating'].value_counts().plot(kind='bar')
    plt.title("Rating Distribution")
    plt.show()

def plot_platform_distribution(df):
    plt.figure(figsize=(10,6))
    df['platform'].value_counts().plot(kind='bar')
    plt.title("Platform Distribution")
    plt.show()

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_test,y_pred,title="Confusion Matrix"):
    labels = sorted(y_test.unique())
    cm=confusion_matrix(y_test,y_pred,labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()