import unittest
import requests
import time
import subprocess
import sys
import chess

class TestChessServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the server
        cls.server_process = subprocess.Popen([sys.executable, "chess_server.py"],
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
        # Wait for the server to start
        time.sleep(10)  # Increased wait time
        cls.base_url = "http://127.0.0.1:5000"

        # Check if the server started successfully
        if cls.server_process.poll() is not None:
            raise RuntimeError("Server failed to start. Check chess_server.py for errors.")

    @classmethod
    def tearDownClass(cls):
        # Stop the server
        cls.server_process.terminate()
        cls.server_process.wait()

        # Print server output for debugging
        stdout, stderr = cls.server_process.communicate()
        print("Server stdout:", stdout.decode())
        print("Server stderr:", stderr.decode())

    def test_init_game(self):
        response = requests.post(f"{self.base_url}/init")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["message"], "New game initialized")

    def test_make_move(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Make a valid move using SAN notation
        response = requests.post(f"{self.base_url}/move", json={"move": "e4"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["message"], "Move e4 applied")

        # Try an invalid move
        response = requests.post(f"{self.base_url}/move", json={"move": "e5e6"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "Invalid move")

    def test_get_ai_move(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Make a move
        requests.post(f"{self.base_url}/move", json={"move": "e4"})
        
        # Get AI's move
        response = requests.get(f"{self.base_url}/get_move")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("move", data)
        self.assertIn("new_fen", data)
        
        # Verify that the returned move is valid
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")  # Board after e4
        try:
            ai_move = board.parse_san(data["move"])
        except ValueError:
            self.fail(f"Invalid move returned: {data['move']}")
        self.assertIn(ai_move, board.legal_moves)

    def test_get_ai_move_game_over(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Play a fool's mate to end the game quickly
        requests.post(f"{self.base_url}/move", json={"move": "f3"})
        requests.post(f"{self.base_url}/move", json={"move": "e5"})
        requests.post(f"{self.base_url}/move", json={"move": "g4"})
        requests.post(f"{self.base_url}/move", json={"move": "Qh4#"})
        
        # Try to get a move when the game is over
        response = requests.get(f"{self.base_url}/get_move")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "game_over")
        self.assertIn("result", data)

    def test_get_board(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Get initial board state
        response = requests.get(f"{self.base_url}/board")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["fen"], "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        # Make a move
        requests.post(f"{self.base_url}/move", json={"move": "e4"})
        
        # Get updated board state
        response = requests.get(f"{self.base_url}/board")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["fen"], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    def test_invalid_requests(self):
        # Test missing move in /move endpoint
        response = requests.post(f"{self.base_url}/move", json={})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)

        # Test invalid JSON in /move endpoint
        response = requests.post(f"{self.base_url}/move", data="invalid json")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)

        # Test invalid endpoint
        response = requests.get(f"{self.base_url}/invalid_endpoint")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
