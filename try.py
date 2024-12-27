import cv2
import numpy as np
import time

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

def check_winner(grid):
    """Checks for a winner in the current grid."""
    # Check rows and columns
    for i in range(3):
        if abs(sum(grid[i])) == 3:  # Row win
            return "X" if grid[i][0] == 1 else "O"
        if abs(sum([grid[j][i] for j in range(3)])) == 3:  # Column win
            return "X" if grid[0][i] == 1 else "O"

    # Check diagonals
    if abs(grid[0][0] + grid[1][1] + grid[2][2]) == 3:  # Main diagonal
        return "X" if grid[0][0] == 1 else "O"
    if abs(grid[0][2] + grid[1][1] + grid[2][0]) == 3:  # Anti-diagonal
        return "X" if grid[0][2] == 1 else "O"

    return None  # No winner yet

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

        # Check for a winner
        winner = check_winner(grid)
        if winner:
            print(f"Winner: {winner}")
            break

        # Print free spaces
        free_spaces = get_free_spaces(board, grid)
        print("Free Spaces:", free_spaces)

        # Check for draw
        if not free_spaces:
            print("It's a draw!")
            break

        time.sleep(2)  # Delay of 2 seconds for better debugging and camera processing

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()