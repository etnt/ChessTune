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
        cls.base_url = "http://localhost:5000"

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

if __name__ == "__main__":
    unittest.main()
