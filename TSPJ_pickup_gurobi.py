import gurobipy as gp
from gurobipy import GRB

def solve_tspj(n, dist_matrix, proc_times):
    """
    Solve TSP-J problem
    """
    
    # Create model
    m = gp.Model("TSPJ")
    
    # Sets
    N = range(n)        # All nodes including depot (0)
    V = range(1, n)     # Cities excluding depot
    J = range(1, n)     # Job types
    
    # Decision Variables
    seq_d = m.addVars(V, len(V), vtype=GRB.BINARY, name="seq_d")  # dropoff sequence
    seq_p = m.addVars(V, len(V), vtype=GRB.BINARY, name="seq_p")  # pickup sequence
    z = m.addVars(V, J, vtype=GRB.BINARY, name="z")      # job assignments
    
    # Time variables
    a = m.addVars(V, vtype=GRB.CONTINUOUS, name="a")    # arrival for dropoff
    c = m.addVars(V, vtype=GRB.CONTINUOUS, name="c")    # job completion time
    s = m.addVars(V, vtype=GRB.CONTINUOUS, name="s")    # arrival for pickup
    last_dropoff = m.addVar(vtype=GRB.CONTINUOUS, name="last_dropoff")
    final_time = m.addVar(vtype=GRB.CONTINUOUS, name="final_time")
    
    # Objective
    m.setObjective(final_time, GRB.MINIMIZE)
    
    # Each city appears exactly once in each sequence
    for i in V:
        m.addConstr(gp.quicksum(seq_d[i,k] for k in range(len(V))) == 1)
        m.addConstr(gp.quicksum(seq_p[i,k] for k in range(len(V))) == 1)
    
    # Each position has exactly one city
    for k in range(len(V)):
        m.addConstr(gp.quicksum(seq_d[i,k] for i in V) == 1)
        m.addConstr(gp.quicksum(seq_p[i,k] for i in V) == 1)
    
    # Job Assignment
    for i in V:
        m.addConstr(gp.quicksum(z[i,j] for j in J) == 1)
    for j in J:
        m.addConstr(gp.quicksum(z[i,j] for i in V) == 1)
    
    # Big M calculation
    Big_M = sum(max(row) for row in dist_matrix) + sum(max(row) for row in proc_times)
    
    # First city timing
    for i in V:
        m.addConstr(a[i] >= dist_matrix[0][i] - Big_M*(1-seq_d[i,0]))
    
    # Subsequent dropoffs
    for k in range(len(V)-1):
        for i in V:
            for j in V:
                if i != j:
                    m.addConstr(a[j] >= a[i] + dist_matrix[i][j] - 
                              Big_M*(2-seq_d[i,k]-seq_d[j,k+1]))
    
    # Track last dropoff
    for i in V:
        m.addConstr(last_dropoff >= a[i])
        m.addConstr(last_dropoff >= a[i] + Big_M*(seq_d[i,len(V)-1] - 1))
    
    # Job completion
    for i in V:
        m.addConstr(c[i] == a[i] + gp.quicksum(proc_times[i][j]*z[i,j] for j in J))
    
    # Pickup timing
    # First pickup must include travel from last dropoff
    for i in V:
        for j in V:
            m.addConstr(s[i] >= last_dropoff + dist_matrix[j][i] - 
                       Big_M*(2-seq_d[j,len(V)-1]-seq_p[i,0]))
            m.addConstr(s[i] >= c[i])  # Can't pickup before completion
    
    # Subsequent pickups
    for k in range(len(V)-1):
        for i in V:
            for j in V:
                if i != j:
                    m.addConstr(s[j] >= s[i] + dist_matrix[i][j] - 
                              Big_M*(2-seq_p[i,k]-seq_p[j,k+1]))
    
    # Final time
    for i in V:
        m.addConstr(final_time >= s[i] + dist_matrix[i][0] - 
                   Big_M*(1-seq_p[i,len(V)-1]))
    
    # Solve with time limit
    m.setParam('TimeLimit', 7200)  # 1 hour time limit
    m.setParam('MIPGap', 0.01)     # 1% gap tolerance
    m.optimize()
    
    # Modified result handling
    if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
        print(f"\nOptimization status: {m.status}")
        print(f"Best objective value: {m.ObjVal:.2f}")
        print(f"Best bound: {m.ObjBound:.2f}")
        print(f"Gap: {100*m.MIPGap:.2f}%")
        print(f"Solution count: {m.SolCount}")
        
        dropoff_seq = []
        for k in range(len(V)):
            for i in V:
                if seq_d[i,k].X > 0.5:
                    dropoff_seq.append(i)
        
        pickup_seq = []
        for k in range(len(V)):
            for i in V:
                if seq_p[i,k].X > 0.5:
                    pickup_seq.append(i)
        
        return {
            'status': m.status,
            'objective': m.ObjVal,
            'best_bound': m.ObjBound,
            'gap': m.MIPGap,
            'dropoff_sequence': dropoff_seq,
            'pickup_sequence': pickup_seq,
            'job_assignments': {(i,j): z[i,j].X for i in V for j in J if z[i,j].X > 0.5},
            'dropoff_times': {i: a[i].X for i in V},
            'completion_times': {i: c[i].X for i in V},
            'pickup_times': {i: s[i].X for i in V},
            'last_dropoff_time': last_dropoff.X
        }
    else:
        print(f"\nOptimization status: {m.status}")
        if m.status == GRB.INFEASIBLE:
            m.computeIIS()
            print("\nConstraints in IIS:")
            for c in m.getConstrs():
                if c.IISConstr:
                    print(f"{c.ConstrName}: {m.getRow(c)} {c.Sense} {c.RHS}")
        return None

if __name__ == "__main__":
    # Read and parse the CSV files
    import os
    
    # Read files
    with open('gr17_TSPJ_TT.csv', 'r') as f:
        tt_content = f.read()
    with open('gr17_TSPJ_JT.csv', 'r') as f:
        jt_content = f.read()
    
    # Parse matrices
    def parse_csv(content):
        rows = content.strip().split('\n')
        matrix = []
        for row in rows:
            matrix.append([int(x) for x in row.split(',')])
        return matrix
    
    dist_matrix = parse_csv(tt_content)
    proc_times = parse_csv(jt_content)
    n = len(dist_matrix)
    
    print(f"Problem size: {n}x{n}")
    
    result = solve_tspj(n, dist_matrix, proc_times)
    
    if result:
        print("\nSolution details:")
        print(f"Total time: {result['objective']:.2f}")
        print(f"Best bound: {result['best_bound']:.2f}")
        print(f"Gap: {100*result['gap']:.2f}%")
        
        print("\nDropoff sequence:")
        print(f"0 -> {' -> '.join(map(str, result['dropoff_sequence']))}")
        
        print("\nDetailed timeline:")
        
        # Verify dropoff times
        print("\nVerified dropoff times:")
        current_time = 0
        current_loc = 0
        for next_city in result['dropoff_sequence']:
            travel_time = dist_matrix[current_loc][next_city]
            current_time += travel_time
            print(f"Travel from {current_loc} to {next_city}: +{travel_time} = {current_time}")
            current_loc = next_city
        
        for i in sorted(result['dropoff_times'].keys()):
            city = i
            dropoff = result['dropoff_times'][i]
            job = [j for (c,j) in result['job_assignments'].keys() if c == i][0]
            proc_time = proc_times[i][job]
            complete = result['completion_times'][i]
            pickup = result['pickup_times'][i]
            
            print(f"\nCity {city}:")
            print(f"  Assigned Job: {job}")
            print(f"  Dropoff at: {dropoff:.2f}")
            print(f"  Processing time: {proc_time}")
            print(f"  Job completes at: {complete:.2f}")
            print(f"  Pickup at: {pickup:.2f}")
            
        print("\nPickup sequence:")
        print(f"{' -> '.join(map(str, result['pickup_sequence']))} -> 0")
        
        # Verify final return to depot
        last_pickup = result['pickup_sequence'][-1]
        last_pickup_time = result['pickup_times'][last_pickup]
        time_to_depot = dist_matrix[last_pickup][0]
        print(f"\nReturn to depot:")
        print(f"Last pickup at city {last_pickup} at time: {last_pickup_time:.2f}")
        print(f"Travel time to depot: {time_to_depot}")
        print(f"Final arrival at depot: {result['objective']:.2f}")
    else:
        print("\nNo solution found")