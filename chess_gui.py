import pygame
import chess
import requests
import sys
from requests.exceptions import RequestException

API_URL = "http://localhost:9999"  # Update this with your server's URL

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 600, 650  # Increased height to accommodate buttons
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game vs AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)

# Load chess piece images
pieces = {}
for color in ['w', 'b']:
    for piece in ['p', 'r', 'n', 'b', 'q', 'k']:
        image = pygame.image.load(f"chess_pieces/{color}{piece}.png")
        pieces[piece.upper() if color == 'w' else piece.lower()] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else GRAY
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = pieces.get(piece.symbol())
                if piece_image:
                    screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    print(f"Warning: Image for piece '{piece.symbol()}' not found")

def get_square_from_mouse(pos):
    x, y = pos
    return chess.square(x // SQUARE_SIZE, 7 - (y // SQUARE_SIZE))

def draw_button(screen, text, position, width, height, color, text_color):
    pygame.draw.rect(screen, color, (*position, width, height))
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(position[0] + width // 2, position[1] + height // 2))
    screen.blit(text_surface, text_rect)

def promote_pawn():
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    promotion_images = [pieces[chess.PIECE_SYMBOLS[piece].upper()] for piece in promotion_pieces]
    
    promotion_rects = []
    for i, image in enumerate(promotion_images):
        rect = image.get_rect()
        rect.centerx = WIDTH // 2
        rect.centery = HEIGHT // 2 - 75 + i * 50
        promotion_rects.append(rect)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for i, rect in enumerate(promotion_rects):
                    if rect.collidepoint(pos):
                        return promotion_pieces[i]
        
        screen.fill(WHITE)
        for i, (image, rect) in enumerate(zip(promotion_images, promotion_rects)):
            screen.blit(image, rect)
        pygame.display.flip()

def main():
    board = chess.Board()
    selected_square = None
    status_message = "Your turn"

    # Define button dimensions and positions
    button_width = 100
    button_height = 40
    undo_button = pygame.Rect(10, HEIGHT - 50, button_width, button_height)
    redo_button = pygame.Rect(WIDTH - button_width - 10, HEIGHT - 50, button_width, button_height)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if undo_button.collidepoint(pos):
                    try:
                        response = requests.post(f"{API_URL}/undo")
                        response.raise_for_status()
                        response_data = response.json()
                        if response_data["status"] == "ok":
                            board = chess.Board(response_data["fen"])
                            status_message = "Move undone"
                        else:
                            status_message = response_data.get("message", "Error undoing move")
                    except RequestException as e:
                        status_message = f"Connection error: {str(e)}"
                
                elif redo_button.collidepoint(pos):
                    try:
                        response = requests.post(f"{API_URL}/redo")
                        response.raise_for_status()
                        response_data = response.json()
                        if response_data["status"] == "ok":
                            board = chess.Board(response_data["fen"])
                            status_message = "Move redone"
                        else:
                            status_message = response_data.get("message", "Error redoing move")
                    except RequestException as e:
                        status_message = f"Connection error: {str(e)}"
                
                else:
                    clicked_square = get_square_from_mouse(pos)
                    
                    if selected_square is None:
                        selected_square = clicked_square
                    else:
                        move = chess.Move(selected_square, clicked_square)
                        promotion = None
                        if board.piece_at(selected_square) == chess.Piece(chess.PAWN, board.turn) and chess.square_rank(clicked_square) in [0, 7]:
                            promotion = promote_pawn()
                            if promotion:
                                move = chess.Move(selected_square, clicked_square, promotion=promotion)
                        
                        if move in board.legal_moves:
                            try:
                                print(f"Sending move to server: {move.uci()}")
                                response = requests.post(f"{API_URL}/move", json={"move": move.uci(), "promotion": chess.PIECE_SYMBOLS[promotion] if promotion else None}, headers={"Content-Type": "application/json"})
                                print(f"Server response status code: {response.status_code}")
                                print(f"Server response headers: {response.headers}")
                                print(f"Server response content: {response.text}")
                                response.raise_for_status()
                                response_data = response.json()
                                print(f"Server response data: {response_data}")
                                if response_data["status"] == "ok":
                                    board.push(move)
                                    status_message = "AI is thinking..."
                                    
                                    print("Requesting AI move")
                                    # Get AI move
                                    ai_response = requests.get(f"{API_URL}/get_move", headers={"Content-Type": "application/json"})
                                    print(f"AI response status code: {ai_response.status_code}")
                                    print(f"AI response headers: {ai_response.headers}")
                                    print(f"AI response content: {ai_response.text}")
                                    ai_response.raise_for_status()
                                    ai_data = ai_response.json()
                                    print(f"AI response data: {ai_data}")
                                    if ai_data["status"] == "ok":
                                        ai_move = chess.Move.from_uci(ai_data["move"])
                                        board.push(ai_move)
                                        status_message = f"AI moved: {ai_move.uci()}"
                                    elif ai_data["status"] == "game_over":
                                        status_message = f"Game over. Result: {ai_data['result']}"
                                    else:
                                        status_message = f"Error getting AI move: {ai_data.get('message', 'Unknown error')}"
                                else:
                                    status_message = f"Error applying move: {response_data.get('message', 'Unknown error')}"
                            except RequestException as e:
                                print(f"Connection error: {str(e)}")
                                print(f"Error details: {e.response.text if e.response else 'No response'}")
                                print(f"Full exception: {repr(e)}")
                                status_message = f"Connection error: {str(e)}"
                        else:
                            status_message = "Invalid move"
                        selected_square = None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:  # New game
                    try:
                        response = requests.post(f"{API_URL}/init")
                        response.raise_for_status()
                        if response.json()["status"] == "ok":
                            board = chess.Board()
                            status_message = "New game started"
                        else:
                            status_message = "Error starting new game"
                    except RequestException as e:
                        status_message = f"Connection error: {str(e)}"

        screen.fill(WHITE)
        draw_board(board)
        
        # Draw undo and redo buttons
        draw_button(screen, "Undo", (undo_button.x, undo_button.y), button_width, button_height, LIGHT_BLUE, BLACK)
        draw_button(screen, "Redo", (redo_button.x, redo_button.y), button_width, button_height, LIGHT_BLUE, BLACK)
        
        # Draw status message
        font = pygame.font.Font(None, 36)
        text = font.render(status_message, True, BLACK)
        screen.blit(text, (10, HEIGHT - 90))

        pygame.display.flip()

if __name__ == "__main__":
    main()
