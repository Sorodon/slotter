import argparse
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpBinary, value
import pandas as pd
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import statistics

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Default log level

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Default log level for the handler

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

def read_preferences(file_path):
    """
    Reads the preferences from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing a list of person names and a preference matrix (2D list).
    """
    logger.info(f"Reading preferences from {file_path}")
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Check if the file is empty
        if data.empty:
            logger.error("The CSV file is empty.")
            raise ValueError("The CSV file is empty.")

        # Check if the first column (names) and at least one timeslot column exist
        if data.shape[1] < 2:
            logger.error("The CSV file must have at least one column for names and one column for timeslots.")
            raise ValueError("The CSV file must have at least one column for names and one column for timeslots.")

        # Check for missing values in the timeslot columns
        if data.iloc[:, 1:].isnull().values.any():
            logger.warning("Missing values detected in the timeslot columns. Filling them with a high penalty value (1e6).")
            data.iloc[:, 1:] = data.iloc[:, 1:].fillna(1e6)  # Fill missing values with a high penalty value

        # Extract person names and preference matrix
        person_names = data.iloc[:, 0].tolist()  # First column contains person names
        preference_matrix = data.iloc[:, 1:].values  # Remaining columns are the preference matrix

        # Ensure all preference values are numeric
        if not pd.api.types.is_numeric_dtype(data.iloc[:, 1:].stack()):
            logger.error("All timeslot preference values must be numeric.")
            raise ValueError("All timeslot preference values must be numeric.")

        return person_names, preference_matrix

    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty or improperly formatted.")
        raise ValueError("The CSV file is empty or improperly formatted.")
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        raise ValueError(f"An error occurred while reading the CSV file: {e}")

def assign_slots(person_names, preference_matrix, min_slots=1, max_slots=2):
    """
    Assign timeslots to people based on preferences.

    Parameters:
        person_names (list): List of person names.
        preference_matrix (2D list): Preference matrix with scores for each person and timeslot.
        min_slots (int): Minimum number of slots each person should receive.
        max_slots (int): Maximum number of slots each person should receive.

    Returns:
        tuple: A tuple containing:
            - A list of tuples with timeslot and assigned person names (sorted by timeslot).
            - A dictionary with statistics about the assignment.
    """
    num_people, num_slots = len(person_names), len(preference_matrix[0])

    # Create the optimization problem
    problem = LpProblem("Slot_Assignment", LpMinimize)

    # Create binary variables
    x = LpVariable.dicts("x", (range(num_people), range(num_slots)), cat=LpBinary)

    # Objective function: Minimize total preference score
    problem += lpSum(preference_matrix[i][j] * x[i][j] for i in range(num_people) for j in range(num_slots))

    # Constraints: Each person gets at least `min_slots` and at most `max_slots`
    for i in range(num_people):
        problem += lpSum(x[i][j] for j in range(num_slots)) >= min_slots  # At least `min_slots`
        problem += lpSum(x[i][j] for j in range(num_slots)) <= max_slots  # At most `max_slots`

    # Constraints: Each timeslot is assigned to exactly one person
    for j in range(num_slots):
        problem += lpSum(x[i][j] for i in range(num_people)) == 1  # Exactly 1 person per slot

    # Capture solver output
    logger.info("Problem setup complete. Solving...")
    solver_output = io.StringIO()
    with redirect_stdout(solver_output), redirect_stderr(solver_output):
        problem.solve()

    # Log the solver output at INFO level
    logger.info(solver_output.getvalue())

    # Check solver status
    if LpStatus[problem.status] != "Optimal":
        logger.error("Solver failed to find an optimal solution.")
        return [], {}
    else:
        # Output the assignments
        assignments = []
        for i in range(num_people):
            for j in range(num_slots):
                if value(x[i][j]) == 1:
                    assignments.append((j, person_names[i], preference_matrix[i][j]))

        # Sort assignments by timeslot
        assignments.sort(key=lambda x: x[0])

        # Calculate statistics
        assigned_preferences = [pref for _, _, pref in assignments]
        best_preference = min(assigned_preferences)
        worst_preference = max(assigned_preferences)
        average_preference = sum(assigned_preferences) / len(assigned_preferences)
        median_preference = statistics.median(assigned_preferences)

        # Overview of slots by average preference
        slot_preferences = {slot: [] for slot in range(num_slots)}
        for slot, person, pref in assignments:
            slot_preferences[slot].append(pref)

        slot_overview = {
            slot: {
                "average": sum(prefs) / len(prefs),
                "median": statistics.median(prefs),
            }
            for slot, prefs in slot_preferences.items()
        }

        stats = {
            "best_assigned_preference": best_preference,
            "worst_assigned_preference": worst_preference,
            "average_assigned_preference": average_preference,
            "median_assigned_preference": median_preference,
            "total_score": sum(assigned_preferences),
            "slot_overview": slot_overview,
        }

        return assignments, stats


def main():
    """
    Main function to handle CLI arguments, read preferences, assign slots, and print results.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Assign timeslots to people based on preferences.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing preferences.")
    parser.add_argument("-m", "--min-slots", type=int, default=1, help="Minimum number of slots each person should receive (default: 1).")
    parser.add_argument("-M", "--max-slots", type=int, default=2, help="Maximum number of slots each person should receive (default: 2).")
    parser.add_argument("-l", "--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: WARNING).")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(args.log_level.upper())
    console_handler.setLevel(args.log_level.upper())

    # Read preferences and assign slots
    person_names, preference_matrix = read_preferences(args.file_path)
    assignments, stats = assign_slots(person_names, preference_matrix, args.min_slots, args.max_slots)

    # Print the results sorted by timeslot
    print("\nAssignments (sorted by timeslot):")
    for timeslot, person, preference in assignments:
        print(f"Timeslot {timeslot}: {person} (Preference: {preference})")

    # Print statistics
    print("\nStatistics:")
    print(f"Best Assigned Preference: {stats['best_assigned_preference']}")
    print(f"Worst Assigned Preference: {stats['worst_assigned_preference']}")
    print(f"Average Assigned Preference: {stats['average_assigned_preference']:.2f}")
    print(f"Median Assigned Preference: {stats['median_assigned_preference']:.2f}")
    print(f"Total Score: {stats['total_score']}")

    # Print overview of slots by average preference
    print("\nOverview of Slots by Average Preference:")
    for slot, overview in stats["slot_overview"].items():
        print(f"Timeslot {slot}: Average = {overview['average']:.2f}, Median = {overview['median']:.2f}")


if __name__ == "__main__":
    main()
