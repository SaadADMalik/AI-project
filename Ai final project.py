import json
import os
import pygame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy




# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Load songs metadata from a JSON file
def load_songs():
    try:
        with open('songs.json', 'r', encoding='utf-8') as file:
            return json.load(file)  # Load JSON data
    except FileNotFoundError:
        print("Error: 'songs.json' file not found!")
        return []
    except json.JSONDecodeError:
        print("Error: 'songs.json' contains invalid JSON!")
        return []

# Display the list of songs
def display_songs(songs):
    print("\nAvailable Songs:")
    for song in songs:
        print(f"{song['id']}. {song['title']} - {song['artist']} ({song['genre']})")

# Recommend songs using content-based filtering
def predictive_recommendations(songs, liked_song_id):
    liked_song = next((song for song in songs if song['id'] == liked_song_id), None)
    if not liked_song:
        print("Song not found!")
        return []

    # Combine metadata for similarity comparison
    song_data = [f"{song['genre']} {song['artist']} {song['title']}" for song in songs]

    # Calculate similarity
    vectorizer = CountVectorizer().fit_transform(song_data)
    similarity_matrix = cosine_similarity(vectorizer)

    # Get recommendations
    liked_index = songs.index(liked_song)
    similarity_scores = list(enumerate(similarity_matrix[liked_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Exclude the liked song itself and return top recommendations
    recommended_indices = [index for index, score in sorted_scores if index != liked_index][:5]
    return [songs[index] for index in recommended_indices]

# NLP-based song search
def nlp_search(songs, query):
    doc = nlp(query.lower())
    genre, artist = None, None

    # Extract genre and artist information from the query
    for token in doc:
        if token.text in [song['genre'].lower() for song in songs]:
            genre = token.text
        if token.text in [song['artist'].lower() for song in songs]:
            artist = token.text

    # Filter songs based on extracted information
    filtered_songs = [
        song for song in songs
        if (genre and song['genre'].lower() == genre) or (artist and song['artist'].lower() == artist)
    ]

    return filtered_songs

# Play a song
def play_song(song_file):
    if not os.path.exists(song_file):
        print(f"Error: File '{song_file}' not found!")
        return

    pygame.mixer.init()
    pygame.mixer.music.load(song_file)
    pygame.mixer.music.play()
    print(f"Playing '{song_file}'... Press 'q' to quit the song.")

    while pygame.mixer.music.get_busy():
        command = input("> ")
        if command.lower() == 'q':
            pygame.mixer.music.stop()
            break

# Main function
def main():
    songs = load_songs()
    if not songs:
        print("No songs available. Exiting...")
        return

    display_songs(songs)

    while True:
        print("\nMenu:")
        print("1. Play a song")
        print("2. Get recommendations")
        print("3. NLP Search")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            try:
                song_name = input("Enter the exact song name to play: ").strip()
                song = next((s for s in songs if s['title'].lower() == song_name.lower()), None)
                if song:
                    play_song(song['file'])
                else:
                    print("Song not found!")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '2':
            try:
                liked_song_id = int(input("Enter a song ID you liked: "))
                recommendations = predictive_recommendations(songs, liked_song_id)
                if recommendations:
                    print("\nRecommended Songs:")
                    for song in recommendations:
                        print(f"{song['id']}. {song['title']} - {song['artist']}")
                else:
                    print("No recommendations found!")
            except ValueError:
                print("Invalid input! Please enter a valid song ID.")
        elif choice == '3':
            query = input("Enter your query (e.g., 'Play some pop music by Adele'): ")
            results = nlp_search(songs, query)
            if results:
                print("\nSearch Results:")
                for song in results:
                    print(f"{song['id']}. {song['title']} - {song['artist']}")
            else:
                print("No songs matched your query.")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()

