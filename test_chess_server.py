import unittest
import requests
import time
import subprocess
import sys

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
        
        # Make a valid move
        response = requests.post(f"{self.base_url}/move", json={"move": "e2e4"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["message"], "Move e2e4 applied")

        # Try an invalid move
        response = requests.post(f"{self.base_url}/move", json={"move": "e4e3"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "Invalid move")

    def test_get_ai_move(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Make a move
        requests.post(f"{self.base_url}/move", json={"move": "e2e4"})
        
        # Get AI's move
        response = requests.get(f"{self.base_url}/get_move")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("move", data)

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
        requests.post(f"{self.base_url}/move", json={"move": "e2e4"})
        
        # Get updated board state
        response = requests.get(f"{self.base_url}/board")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["fen"], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    def test_make_ai_move(self):
        # Initialize a new game
        requests.post(f"{self.base_url}/init")
        
        # Test with a valid FEN string
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        response = requests.post(f"{self.base_url}/make_ai_move", json={"fen": initial_fen})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("move", data)
        self.assertIn("new_fen", data)
        self.assertNotEqual(data["new_fen"], initial_fen)

        # Test with an invalid FEN string
        invalid_fen = "invalid_fen_string"
        response = requests.post(f"{self.base_url}/make_ai_move", json={"fen": invalid_fen})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "Invalid FEN string")

        # Test with a game-over position
        game_over_fen = "4k3/4P3/4K3/8/8/8/8/8 b - - 0 1"  # Black is in checkmate
        response = requests.post(f"{self.base_url}/make_ai_move", json={"fen": game_over_fen})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "game_over")
        self.assertIn("result", data)

        # Test without providing a FEN string
        response = requests.post(f"{self.base_url}/make_ai_move", json={})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "FEN string is required")

if __name__ == "__main__":
    unittest.main()
