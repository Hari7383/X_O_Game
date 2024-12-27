import cv2
import numpy as np
import time
import _socket

def send_data(free_spaces):
    """Sends free space data to the ESP32."""
    ESP32_IP = "192.168.29.76"  # Replace with the ESP32's IP address
    PORT = 8080
    client_socket = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        # Connect to the ESP32
        client_socket.connect((ESP32_IP, PORT))
        print(f"Connected to ESP32 at {ESP32_IP}:{PORT}")
        
        # Format the free spaces as a comma-separated string
        free_spaces_str = ",".join(map(str, free_spaces))
        client_socket.sendall(free_spaces_str.encode('utf-8'))
        print("Sent to ESP32:", free_spaces_str)
              
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

def detect_shape(cell):
    """Detects whether a cell contains an X, O, or is empty."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        
        if len(approx) == 4 and area > 100:  # Detect X (intersecting lines)
            return "X"
        elif len(approx) > 8 and area > 100:  # Detect O (circle-like shape)
            return "O"

    return None  # Empty cell

def update_board(board, grid, i, j, symbol):
    """Updates the board with the symbol at the corresponding cell."""
    if symbol == "X":
        board[i][j] = "X"
    elif symbol == "O":
        board[i][j] = "O"
    grid[i][j] = 1 if symbol == "X" else -1
    return board

def get_free_spaces(board, grid):
    """Returns the coordinates of the free spaces."""
    free_spaces = []
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:  # Free space
                free_spaces.append(board[i][j])  # Add cell value (like 13, 31, etc.)
    return free_spaces

def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Initialize a 3x3 grid representing available spots (0 = Free, 1 = X, -1 = O)
    grid = [[0 for _ in range(3)] for _ in range(3)]  # Free spaces = 0, X = 1, O = -1
    board = [
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33]
    ]

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Define grid coordinates
        h, w, _ = frame.shape
        cell_h, cell_w = h // 3, w // 3

        for i in range(3):
            for j in range(3):
                # Extract each cell
                x1, y1 = j * cell_w, i * cell_h
                x2, y2 = (j + 1) * cell_w, (i + 1) * cell_h
                cell = frame[y1:y2, x1:x2]

                shape = detect_shape(cell)
                if shape == "X" and grid[i][j] == 0:
                    grid[i][j] = 1  # Mark X
                    board = update_board(board, grid, i, j, "X")
                elif shape == "O" and grid[i][j] == 0:
                    grid[i][j] = -1  # Mark O
                    board = update_board(board, grid, i, j, "O")

                # Draw on frame for debugging
                color = (0, 255, 0) if shape == "X" else (0, 0, 255) if shape == "O" else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display the frame with grid overlays
        cv2.imshow("Tic-Tac-Toe Detection", frame)

        # Print updated board
        print("Updated Board:")
        for row in board:
            print(row)

        # Print free spaces and send them to ESP32
        free_spaces = get_free_spaces(board, grid)
        print("Free Spaces:", free_spaces)

        # Send free spaces to ESP32
        if free_spaces:
            send_data(free_spaces)

        time.sleep(2)  # Delay of 2 seconds for better debugging and camera processing

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
