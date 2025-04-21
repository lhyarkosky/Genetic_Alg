import pandas as pd
import time
import numpy as np
import random



def get_rtf(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['Rooms', 'Times', 'Facilitators','Activities']].dropna(how='all')
    rooms = df['Rooms'].dropna().tolist()
    times = df['Times'].dropna().tolist()
    facilitators = df['Facilitators'].dropna().tolist()
    activities = df['Activities'].dropna().tolist()
    return rooms, times, facilitators, activities

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tournament_selection(scored_population, num_selected, tournament_size=3):
    selected = []
    for _ in range(num_selected):
        tournament = random.sample(scored_population, tournament_size)
        best = max(tournament, key=lambda x: x[1])
        selected.append(best[0])
    return selected

def create_random_schedule(rooms: list, times: list, facilitators: list, activities: list) -> pd.DataFrame:
    schedule_data = {
        'Activity': [],
        'Room': [],
        'Time': [],
        'Facilitator': []
    }

    for activity in activities:
        random_room = random.choice(rooms)
        random_time = random.choice(times)
        random_facilitator = random.choice(facilitators)

        schedule_data['Activity'].append(activity)
        schedule_data['Room'].append(random_room)
        schedule_data['Time'].append(random_time)
        schedule_data['Facilitator'].append(random_facilitator)

    return pd.DataFrame(schedule_data)


def fitness(schedule_df: pd.DataFrame, verbose=False):

    room_capacities = {
        "Slater 003": 45, "Roman 216": 30, "Loft 206": 75, "Roman 201": 50,
        "Loft 310": 108, "Beach 201": 60, "Beach 301": 75, "Logos 325": 450, "Frank 119": 60
    }

    expected_enrollments = {
        "SLA100A": 50, "SLA100B": 50, "SLA191A": 50, "SLA191B": 50,
        "SLA201": 50, "SLA291": 50, "SLA303": 60, "SLA304": 25,
        "SLA394": 20, "SLA449": 60, "SLA451": 100
    }

    preferred = {
        "SLA100A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA100B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA201": ["Glen", "Banks", "Zeldin", "Shaw"],
        "SLA291": ["Lock", "Banks", "Zeldin", "Singer"],
        "SLA303": ["Glen", "Zeldin", "Banks"],
        "SLA304": ["Glen", "Banks", "Tyler"],
        "SLA394": ["Tyler", "Singer"],
        "SLA449": ["Tyler", "Singer", "Shaw"],
        "SLA451": ["Tyler", "Singer", "Shaw"],
    }

    alternates = {
        "SLA100A": ["Numen", "Richards"],
        "SLA100B": ["Numen", "Richards"],
        "SLA191A": ["Numen", "Richards"],
        "SLA191B": ["Numen", "Richards"],
        "SLA201": ["Numen", "Richards", "Singer"],
        "SLA291": ["Numen", "Richards", "Shaw", "Tyler"],
        "SLA303": ["Numen", "Singer", "Shaw"],
        "SLA304": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"],
        "SLA394": ["Richards", "Zeldin"],
        "SLA449": ["Zeldin", "Uther"],
        "SLA451": ["Zeldin", "Uther", "Richards", "Banks"]
    }

    fitness_score = 0
    schedule_df = schedule_df.copy()

    # Room-time conflicts
    room_time_conflicts = schedule_df.groupby(['Room', 'Time']).size()
    for (room, time), count in room_time_conflicts.items():
        if count > 1:
            deduction = 0.5 * (count - 1)
            fitness_score -= deduction
            if verbose:
                print(f"Room-time conflict in {room} at {time}: -{deduction:.2f}")

    # Facilitator-time conflicts
    fac_time_conflicts = schedule_df.groupby(['Facilitator', 'Time']).size()
    for (fac, time), count in fac_time_conflicts.items():
        if count > 1:
            deduction = 0.2 * (count - 1)
            fitness_score -= deduction
            if verbose:
                print(f"Facilitator {fac} conflict at {time}: -{deduction:.2f}")
        else:
            # Facilitator has only one activity in this time slot: +0.2
            fitness_score += 0.2
            if verbose:
                print(f"Facilitator {fac} available at {time}: +0.20")

    # Facilitator total load (including those not in the schedule)
    rooms, times, facilitators, activities = get_rtf("input.csv")
    all_facilitators = set(facilitators)  # ensure uniqueness
    
    fac_loads = schedule_df['Facilitator'].value_counts().to_dict()
    
    for fac in all_facilitators:
        count = fac_loads.get(fac, 0)
        
        if fac == "Tyler":
            # No penalty for Tyler with 0 or 1 activities
            if count > 4:
                fitness_score -= 0.5
                if verbose:
                    print(f"Tyler overloaded with {count} sessions: -0.50")
        else:
            # For all other facilitators with at least 1 activity
            if 1 <= count <= 2:  # Penalize facilitators with only 1 or 2 activities
                fitness_score -= 0.4
                if verbose:
                    print(f"{fac} underloaded with {count} sessions: -0.40")
            if count > 4:
                fitness_score -= 0.5
                if verbose:
                    print(f"{fac} overloaded with {count} sessions: -0.50")

    # Activity-specific checks
    for _, row in schedule_df.iterrows():
        activity, room, time, facilitator = row['Activity'], row['Room'], row['Time'], row['Facilitator']
        expected = expected_enrollments.get(activity, 50)
        room_cap = room_capacities.get(room, 0)

        # Room size scoring
        if room_cap < expected:
            fitness_score -= 0.5
            if verbose:
                print(f"{activity} in {room} (cap {room_cap}) too small for {expected} students: -0.50")
        elif room_cap > 6 * expected:
            fitness_score -= 0.4
            if verbose:
                print(f"{activity} in {room} (cap {room_cap}) too large for {expected} students: -0.40")
        elif room_cap > 3 * expected:
            fitness_score -= 0.2
            if verbose:
                print(f"{activity} in {room} (cap {room_cap}) moderately oversized for {expected} students: -0.20")
        else:
            fitness_score += 0.3
            if verbose:
                print(f"{activity} in {room} is appropriately sized: +0.30")

        # Facilitator match scoring
        if facilitator in preferred.get(activity, []):
            fitness_score += 0.5
            if verbose:
                print(f"{facilitator} is a preferred facilitator for {activity}: +0.50")
        elif facilitator in alternates.get(activity, []):
            fitness_score += 0.2
            if verbose:
                print(f"{facilitator} is an alternate facilitator for {activity}: +0.20")
        else:
            fitness_score -= 0.1
            if verbose:
                print(f"{facilitator} is not suitable for {activity}: -0.10")

    # Helper functions for SLA100/SLA191 constraints
    def get_time_numeric(t):
        mapping = {
            "10:00 AM": 10, "11:00 AM": 11, "12:00 PM": 12,
            "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15
        }
        return mapping.get(t, 0)

    def time_diff(t1, t2):
        return abs(get_time_numeric(t1) - get_time_numeric(t2))

    # Get rows for SLA100 and SLA191 sections
    def get_rows(course_prefix):
        if course_prefix == "SLA100":
            return schedule_df[schedule_df['Activity'].isin(["SLA100A", "SLA100B"])].reset_index(drop=True)
        elif course_prefix == "SLA191":
            return schedule_df[schedule_df['Activity'].isin(["SLA191A", "SLA191B"])].reset_index(drop=True)
        else:
            return schedule_df[schedule_df['Activity'].str.startswith(course_prefix)].reset_index(drop=True)

    # Facilitator consecutive activity penalties and bonuses
    facilitator_sessions = schedule_df.groupby('Facilitator')

    for fac, group in facilitator_sessions:
        if len(group) < 2:
            continue

        # Sort sessions by time
        group_sorted = group.copy()
        group_sorted['TimeNumeric'] = group_sorted['Time'].apply(get_time_numeric)
        group_sorted = group_sorted.sort_values('TimeNumeric').reset_index(drop=True)

        for i in range(len(group_sorted) - 1):
            time_diff_consec = group_sorted.loc[i+1, 'TimeNumeric'] - group_sorted.loc[i, 'TimeNumeric']
            if time_diff_consec == 1:  # Consecutive sessions
                # Add +0.5 bonus for consecutive sessions (like SLA100/SLA191 rule)
                fitness_score += 0.5
                if verbose:
                    print(f"{fac} has consecutive sessions: +0.50")
                
                # Check if they're in distant buildings
                room1 = group_sorted.loc[i, 'Room']
                room2 = group_sorted.loc[i+1, 'Room']
                
                is_roman_beach = lambda r: "Roman" in r or "Beach" in r
                
                if is_roman_beach(room1) != is_roman_beach(room2):
                    fitness_score -= 0.4
                    if verbose:
                        print(f"{fac} has back-to-back sessions in distant buildings ({room1} and {room2}): -0.40")

    # Intra-course constraints (SLA100, SLA191)
    for course in ["SLA100", "SLA191"]:
        rows = get_rows(course)
        if len(rows) == 2:
            t1, t2 = rows.loc[0, 'Time'], rows.loc[1, 'Time']
            diff = time_diff(t1, t2)

            if t1 == t2:
                fitness_score -= 0.5
                if verbose:
                    print(f"{course} both sections at same time {t1}: -0.50")
            elif diff >= 4:
                fitness_score += 0.5
                if verbose:
                    print(f"{course} sections separated by {diff} hours: +0.50")

    # Inter-course constraints (between SLA100 and SLA191)
    sla100_rows = get_rows("SLA100")
    sla191_rows = get_rows("SLA191")

    for i in range(len(sla100_rows)):
        for j in range(len(sla191_rows)):
            t1 = sla100_rows.loc[i, 'Time']
            t2 = sla191_rows.loc[j, 'Time']
            r1 = sla100_rows.loc[i, 'Room']
            r2 = sla191_rows.loc[j, 'Room']
            diff = time_diff(t1, t2)

            if diff == 1:
                # Consecutive time slots between SLA100 and SLA191
                fitness_score += 0.5
                if verbose:
                    print(f"SLA100 and SLA191 sessions are consecutive ({t1}, {t2}): +0.50")

                # Penalize if one is in Roman/Beach and the other is not
                roman_beach = lambda r: "Roman" in r or "Beach" in r
                if roman_beach(r1) != roman_beach(r2):
                    fitness_score -= 0.4
                    if verbose:
                        print(f"Consecutive SLA100/SLA191 in distant buildings ({r1}, {r2}): -0.40")

            elif diff == 2:
                # Separated by 1 hour
                fitness_score += 0.25
                if verbose:
                    print(f"SLA100 and SLA191 separated by 1 hour ({t1}, {t2}): +0.25")

            elif t1 == t2:
                # Same time slot
                fitness_score -= 0.25
                if verbose:
                    print(f"SLA100 and SLA191 at same time ({t1}): -0.25")

    return [schedule_df, fitness_score]




def random_breed(population: list) -> list:
    children = []
    while len(children) < len(population):
        parent1, parent2 = random.sample(population, 2)
        child1_data = {'Activity': [], 'Room': [], 'Time': [], 'Facilitator': []}
        child2_data = {'Activity': [], 'Room': [], 'Time': [], 'Facilitator': []}

        for i in range(len(parent1)):
            # Child 1: Randomly select attributes from parents
            room1 = parent1.iloc[i]['Room'] if random.random() < 0.5 else parent2.iloc[i]['Room']
            time1 = parent1.iloc[i]['Time'] if random.random() < 0.5 else parent2.iloc[i]['Time']
            fac1 = parent1.iloc[i]['Facilitator'] if random.random() < 0.5 else parent2.iloc[i]['Facilitator']
            
            # Child 2: Randomly select attributes from parents
            room2 = parent2.iloc[i]['Room'] if random.random() < 0.5 else parent1.iloc[i]['Room']
            time2 = parent2.iloc[i]['Time'] if random.random() < 0.5 else parent1.iloc[i]['Time']
            fac2 = parent2.iloc[i]['Facilitator'] if random.random() < 0.5 else parent1.iloc[i]['Facilitator']
            
            child1_data['Activity'].append(parent1.iloc[i]['Activity'])
            child1_data['Room'].append(room1)
            child1_data['Time'].append(time1)
            child1_data['Facilitator'].append(fac1)

            child2_data['Activity'].append(parent2.iloc[i]['Activity'])
            child2_data['Room'].append(room2)
            child2_data['Time'].append(time2)
            child2_data['Facilitator'].append(fac2)

        children.append(pd.DataFrame(child1_data))
        if len(children) < len(population):
            children.append(pd.DataFrame(child2_data))
    return children


def mutate(mutation_targets: list, mutation_rate: float = 0.2) -> list:
    rooms, times, facilitators, activities = get_rtf("input.csv")
    mutated = []
    for schedule in mutation_targets:
        schedule_copy = schedule.copy()
        for idx in range(len(schedule_copy)):
            # Mutate each field independently
            if random.random() < mutation_rate:
                schedule_copy.at[idx, 'Room'] = random.choice(rooms)
            if random.random() < mutation_rate:
                schedule_copy.at[idx, 'Time'] = random.choice(times)
            if random.random() < mutation_rate:
                schedule_copy.at[idx, 'Facilitator'] = random.choice(facilitators)
        mutated.append(schedule_copy)
    return mutated

def targeted_mutations(schedules, probability=0.3):
    """Apply targeted mutations to address specific constraints"""
    rooms, times, facilitators, activities = get_rtf("input.csv")
    
    # Get dictionaries for preferred facilitators
    preferred = {
        "SLA100A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA100B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA201": ["Glen", "Banks", "Zeldin", "Shaw"],
        "SLA291": ["Lock", "Banks", "Zeldin", "Singer"],
        "SLA303": ["Glen", "Zeldin", "Banks"],
        "SLA304": ["Glen", "Banks", "Tyler"],
        "SLA394": ["Tyler", "Singer"],
        "SLA449": ["Tyler", "Singer", "Shaw"],
        "SLA451": ["Tyler", "Singer", "Shaw"],}
    
    # Get room capacities
    room_capacities = {
        "Slater 003": 45, "Roman 216": 30, "Loft 206": 75, "Roman 201": 50,
        "Loft 310": 108, "Beach 201": 60, "Beach 301": 75, "Logos 325": 450, "Frank 119": 60
    }
    
    # Expected enrollments
    expected_enrollments = {
        "SLA100A": 50, "SLA100B": 50, "SLA191A": 50, "SLA191B": 50,
        "SLA201": 50, "SLA291": 50, "SLA303": 60, "SLA304": 25,
        "SLA394": 20, "SLA449": 60, "SLA451": 100
    }
    
    for schedule in schedules:
        if random.random() < probability:
            # Fix room-time conflicts
            conflicts = schedule.groupby(['Room', 'Time']).size().reset_index(name='count')
            conflicts = conflicts[conflicts['count'] > 1]
            
            if not conflicts.empty:
                conflict = conflicts.sample(1).iloc[0]
                conflict_rows = schedule[(schedule['Room'] == conflict['Room']) & 
                                         (schedule['Time'] == conflict['Time'])]
                
                # Pick a random row to fix and change its time or room
                random_idx = conflict_rows.sample(1).index[0]
                if random.random() < 0.5:
                    schedule.at[random_idx, 'Room'] = random.choice(rooms)
                else:
                    schedule.at[random_idx, 'Time'] = random.choice(times)
        
        # Fix facilitator assignments - assign preferred facilitators
        if random.random() < probability:
            random_idx = random.randrange(len(schedule))
            activity = schedule.iloc[random_idx]['Activity']
            if activity in preferred:
                schedule.at[random_idx, 'Facilitator'] = random.choice(preferred[activity])
        
        # Fix room size issues
        if random.random() < probability:
            for idx, row in schedule.iterrows():
                activity, room = row['Activity'], row['Room']
                expected = expected_enrollments.get(activity, 50)
                room_cap = room_capacities.get(room, 0)
                
                # If room too small or too large, try to fix
                if room_cap < expected or room_cap > 6 * expected:
                    # Find appropriate rooms
                    suitable_rooms = [r for r, cap in room_capacities.items() 
                                     if cap >= expected and cap <= 3 * expected]
                    if suitable_rooms:
                        schedule.at[idx, 'Room'] = random.choice(suitable_rooms)
                    break  # Just fix one issue per schedule
    
    return schedules

def sort(scored_population: list) -> list:
    return sorted(scored_population, key=lambda x: x[1], reverse=True)

def gen_alg(schedules, num_generations: int):
    best_schedule = None
    best_score = -float('inf')
    generations_since_improvement = 0
    start_time = time.time()
    mutation_rate = 0.3  # Higher initial mutation rate
    population_size = len(schedules)
    
    # Parameters for enhanced diversity
    diversity_percentage = 0.1  # 10% of population will be random
    restart_threshold = 15  # Restart after this many generations without improvement
    
    for generation in range(num_generations):
        gen_start = time.time()

        # Score population
        scored_schedules = [fitness(schedule) for schedule in schedules]
        sorted_schedules = sort(scored_schedules)
        current_best = sorted_schedules[0][1]
        current_best_schedule = sorted_schedules[0][0]
        avg_fitness = np.mean([score for _, score in scored_schedules])
        worst_fitness = sorted_schedules[-1][1]

        # Track best solution
        if current_best > best_score:
            best_score = current_best
            best_schedule = current_best_schedule.copy()
            generations_since_improvement = 0
            mutation_rate = max(0.1, mutation_rate * 0.95)  # Reduce mutation when improving
        else:
            generations_since_improvement += 1
            # More aggressive mutation rate increase
            mutation_rate = min(0.6, mutation_rate * 1.2)
        
        # Check for restart condition
        if generations_since_improvement >= restart_threshold:
            print(f"*** Restarting at generation {generation+1} ***")
            # Keep top 20% of solutions
            elite_count = int(0.2 * population_size)
            elite_solutions = [s[0] for s in sorted_schedules[:elite_count]]
            
            # Generate fresh solutions for the rest
            rooms, times, facilitators, activities = get_rtf("input.csv")
            fresh_solutions = [create_random_schedule(rooms, times, facilitators, activities) 
                              for _ in range(population_size - elite_count)]
            
            # Add back the best solution ever found
            if best_schedule is not None:
                fresh_solutions[0] = best_schedule.copy()
                
            schedules = elite_solutions + fresh_solutions
            generations_since_improvement = 0
            mutation_rate=.3
            continue

        # Selection with diversity preservation
        midpoint = len(sorted_schedules) // 2
        best_solutions = tournament_selection(scored_schedules, midpoint - int(diversity_percentage * population_size))
        
        # Add some random solutions for diversity
        rooms, times, facilitators, activities = get_rtf("input.csv")
        random_solutions = [create_random_schedule(rooms, times, facilitators, activities) 
                           for _ in range(int(diversity_percentage * population_size))]
        best_solutions.extend(random_solutions)
        
        # Breeding and mutation
        children = random_breed(best_solutions)
        #children = mutate(children, mutation_rate)
        
        # Add targeted mutations to address common constraint violations
        children = targeted_mutations(children, 0.3)
        
        # New generation - elitism (keep the best solutions)
        elite_count = max(1, int(0.1 * population_size))  # At least 1, up to 10%
        schedules = children + [s[0] for s in sorted_schedules[:elite_count]]
        
        # Ensure we maintain population size
        while len(schedules) > population_size:
            schedules.pop()

        # Progress reporting
        gen_end = time.time()
        print(f"Gen {generation+1} | Best: {current_best:.2f} | Avg: {avg_fitness:.2f} | Worst: {worst_fitness:.2f} | Mut: {mutation_rate:.2f} | Time: {gen_end-gen_start:.2f}s")

    if best_schedule is None:
        # If we never found a better solution than initial
        final_scores = [fitness(s) for s in schedules]
        best_result = max(final_scores, key=lambda x: x[1])
        print("\nBest Score:", best_result[1])
        return best_result[0]
    else:
        # Return the best solution found across all generations
        best_final_score = fitness(best_schedule)[1]
        print("\nBest Score:", best_final_score)
        return best_schedule
    
def save_as_csv(schedule_df, filename="final_schedule.csv"):
    """
    Save the final schedule DataFrame to a CSV file.
    
    Parameters:
    - schedule_df (pd.DataFrame): The final schedule to be saved.
    - filename (str): The output filename (default is 'final_schedule.csv').
    """
    try:
        # Ensure we're saving only relevant columns in a clean format
        cols_to_save = ['Activity', 'Room', 'Time', 'Facilitator']
        output_df = schedule_df[cols_to_save].copy()

        # Save to CSV
        output_df.to_csv(filename, index=False)
        print(f"✅ Schedule saved successfully to '{filename}'")
    except Exception as e:
        print(f"❌ Failed to save schedule: {e}")