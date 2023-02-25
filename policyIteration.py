import random
import wandb

# Arguments
REWARD = 0 
DISCOUNT = 0.99
MAX_ERROR = 10**(-3)

# Set up 
NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
NUM_ROW = 4
NUM_COL = 4
U = [[0, 0, 0, 0],[-10, 0, -10, +10],[0, -10, 0, 0],[0, 0, 0, -10]]
policy = [[random.randint(0, 3) for j in range(NUM_COL)] for i in range(NUM_ROW)] # random policy

# Logging
config = {
    "iteration_type" : "Policy Iteration",
    "environment_name" : "Intergalactic Amnesia"
}
run = wandb.init(
    project = "fundamentals_rl",
    config = config,
    save_code = True,
)

# Visualization
def printEnvironment(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if r == 1 and c == 0:
                val = "-10" 
            elif r == 2 and c == 1:
                val = "-10" 
            elif r == 1 and c == 3:
                val = "+10"
            elif r == 1 and c == 2:
                val = "-10"
            elif r == 3 and c == 3:
                val = "-10"
            else:
                val = ["Down", "Left", "Up", "Right"][arr[r][c]]
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# getU function
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL: # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# calculateU function
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

# Evaluate the policy
def policyEvaluation(policy, U):
    while True:
        nextU = [[0, 0, 0, 0],[-10, 0, -10, +10],[0, -10, 0, 0],[0, 0, 0, -10]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if r == 1 and c == 0:
                    continue
                elif r == 2 and c == 1:
                    continue 
                elif r == 1 and c == 3:
                    continue
                elif r == 1 and c == 2:
                    continue
                elif r == 3 and c == 3:
                    continue
                nextU[r][c] = calculateU(U, r, c, policy[r][c]) # simplified Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        wandb.log({"Position_1" : U[3][0],"Position_2" : U[3][1],"Position_3" : U[3][2],"Position_4" : U[3][3],
                    "Position_5" : U[2][0],"Position_6" : U[2][1],"Position_7" : U[2][2],"Position_8" : U[2][3],
                    "Position_9" : U[1][0],"Position_10" : U[1][1],"Position_11" : U[1][2],"Position_12" : U[1][3],
                    "Position_13" : U[0][0],"Position_14" : U[0][1],"Position_15" : U[0][2],"Position_16" : U[0][3]})
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

def policyIteration(policy, U):
    print("During the policy iteration:\n")
    while True:
        U = policyEvaluation(policy, U)
        unchanged = True
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if r == 1 and c == 0:
                    continue
                elif r == 2 and c == 1:
                    continue 
                elif r == 1 and c == 3:
                    continue
                elif r == 1 and c == 2:
                    continue
                elif r == 3 and c == 3:
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculateU(U, r, c, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculateU(U, r, c, policy[r][c]):
                    policy[r][c] = maxAction # the action that maximizes the utility
                    unchanged = False
        if unchanged:
            break
        printEnvironment(policy)
    return policy

print("The initial random policy is:\n")
printEnvironment(policy)

# Policy iteration
policy = policyIteration(policy, U)

# Optimal policy
print("The optimal policy is:\n")
printEnvironment(policy)