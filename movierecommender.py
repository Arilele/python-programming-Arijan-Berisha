import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

def load_data():
    """Load movies, tags, and ratings data from CSV files"""
    try:
        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        movies_path = current_dir / 'movies.csv'
        tags_path = current_dir / 'tags.csv'
        ratings_path = current_dir / 'ratings.csv'
        
        movies = pd.read_csv(movies_path)
        tags = pd.read_csv(tags_path)
        
        # Bara ta upp viktiga kolumner för att spara minne
        ratings = pd.read_csv(ratings_path, usecols=['movieId', 'rating'])
        
        return movies, tags, ratings
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()

def preprocess_data(movies, tags, ratings):
    """Combine and preprocess movie data"""
    try:
        # Kombinera alla taggar för varje film
        movie_tags = tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(str(t) for t in x if pd.notna(t))
        ).reset_index(name='combined_tags')
        
        # Kalkulera average rating för varje film
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
        
        # Merge med movie data
        movies_merged = pd.merge(movies, movie_tags, on='movieId', how='left')
        movies_merged = pd.merge(movies_merged, avg_ratings, on='movieId', how='left')
        
        movies_merged['combined_tags'] = movies_merged['combined_tags'].fillna('')
        movies_merged['avg_rating'] = movies_merged['avg_rating'].fillna('No ratings')
        
        # Kombinera genrer och taggar till en singel feature
        movies_merged['features'] = movies_merged['genres'] + ' ' + movies_merged['combined_tags']
        
        # Skapa TF-IDF feature matrix
        tfidf = TfidfVectorizer(stop_words='english')
        features_matrix = tfidf.fit_transform(movies_merged['features'])
        
        return movies_merged, features_matrix
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        exit()

def train_model(features_matrix):
    """Train KNN model on feature matrix"""
    try:
        knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 neighbors (1 är self)
        knn.fit(features_matrix)
        return knn
    except Exception as e:
        print(f"Error training model: {str(e)}")
        exit()

def get_movie_selection(movies_merged, search_term):
    """Handle movie selection with multiple matches"""
    # Case-insensitive search
    matches = movies_merged[movies_merged['title'].str.contains(search_term, case=False)]
    
    if len(matches) == 0:
        return None, "Movie not found in database. Please try another title."
    elif len(matches) == 1:
        return matches.index[0], None
    else:
        print("\nMultiple matches found. Please select which movie you meant:")
        for i, (idx, row) in enumerate(matches.iterrows(), 1):
            print(f"{i}. {row['title']} ({row['genres']})")
        
        while True:
            try:
                choice = int(input(f"\nEnter your choice (1-{len(matches)}): "))
                if 1 <= choice <= len(matches):
                    return matches.iloc[choice-1].name, None
                print(f"Please enter a number between 1 and {len(matches)}")
            except ValueError:
                print("Please enter a valid number")

def get_recommendations(movie_idx, movies_merged, knn, features_matrix):
    """Get top 5 similar movies"""
    try:
        # Hitta nearest neighbors (exkludera self)
        distances, indices = knn.kneighbors(features_matrix[movie_idx])
        recommendations = movies_merged.iloc[indices[0][1:6]][['title', 'genres', 'avg_rating']]
        return recommendations, None
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

def main():
    print("Loading data...")
    movies, tags, ratings = load_data()
    
    print("Preprocessing data...")
    movies_merged, features_matrix = preprocess_data(movies, tags, ratings)
    
    print("Training model...")
    knn = train_model(features_matrix)
    
    print("\nMovie Recommendation System (Content-Based)")
    print("----------------------------------------")
    print("This system recommends movies based on genre and tag similarity.\n")
    
    while True:
        search_term = input("\nEnter a movie title (or 'quit' to exit): ").strip()
        
        if search_term.lower() == 'quit':
            break
            
        if not search_term:
            print("Please enter a movie title")
            continue
            
        # Få användarens movie selection
        movie_idx, error = get_movie_selection(movies_merged, search_term)
        
        if error:
            print(f"\nError: {error}")
            continue
            
        # Få och visa rekommendationer
        original_title = movies_merged.loc[movie_idx, 'title']
        original_rating = movies_merged.loc[movie_idx, 'avg_rating']
        
        print(f"\nSelected movie: '{original_title}'")
        print(f"Average rating: {original_rating if isinstance(original_rating, str) else round(original_rating, 2)}/5")
        
        recommendations, error = get_recommendations(movie_idx, movies_merged, knn, features_matrix)
        
        if error:
            print(f"\nError: {error}")
        elif recommendations.empty:
            print(f"\nNo recommendations found for '{original_title}'. Try a different movie.")
        else:
            print(f"\nTop 5 movies similar to '{original_title}':")
            print("----------------------------------------")
            
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                rating = row['avg_rating'] if isinstance(row['avg_rating'], str) else round(row['avg_rating'], 2)
                print(f"\n{i}. {row['title']}")
                print(f"   Genres: {row['genres'].replace('|', ', ')}")
                print(f"   Average rating: {rating}/5")

if __name__ == "__main__":
    main()