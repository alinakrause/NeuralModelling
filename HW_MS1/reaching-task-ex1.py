import pygame
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Game parameters
SCREEN_X, SCREEN_Y = 2560,1600  # your screen resolution
WIDTH, HEIGHT = SCREEN_X // 1  , SCREEN_Y // 1 # be aware of monitor scaling on windows (150%)
CIRCLE_SIZE = 20
TARGET_SIZE = CIRCLE_SIZE
TARGET_RADIUS = 300
MASK_RADIUS = 0.75 * TARGET_RADIUS
ATTEMPTS_LIMIT = 80#200
START_POSITION = (WIDTH // 2, HEIGHT // 2)
START_ANGLE = 0
PERTUBATION_ANGLE= 30
TIME_LIMIT = 1000 # time limit in ms

trial_count = 0
DESIGN_CHANGE = np.asarray([40,80,120,160])
GRAD_START = DESIGN_CHANGE[0]
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Reaching Game")


# Initialize game metrics
score = 0
attempts = 0
new_target = None
start_time = 0
shift_target = False
new_target = None
start_target=math.radians(START_ANGLE)
move_faster = False 
clock = pygame.time.Clock()

# Initialize game modes
mask_mode= True
target_mode = 'fix'  # Mode for angular shift of target: random, fix, dynamic
pertubation_mode= False
pertubation_type= 'sudden' # Mode for angular shift of controll: random, gradual or sudden
perturbation_angle = math.radians(PERTUBATION_ANGLE)  # Angle between mouse_pos and circle_pos

error_angles = []  # List to store error angles


# Function to generate a new target position
def generate_target_position():
    if target_mode == 'random':
        angle = random.uniform(0, 2 * math.pi)

    elif target_mode == 'fix':   
        angle=start_target;  

    new_target_x = WIDTH // 2 + TARGET_RADIUS * math.sin(angle)
    new_target_y = HEIGHT // 2 + TARGET_RADIUS * -math.cos(angle) # zero-angle at the top
    return [new_target_x, new_target_y]

# Function to check if the current target is reached
def check_target_reached():
    if new_target:
        distance = math.hypot(circle_pos[0] - new_target[0], circle_pos[1] - new_target[1])
        return distance <= CIRCLE_SIZE // 2
    return False

# Function to check if player is at starting position and generate new target
def at_start_position_and_generate_target(mouse_pos):
    distance = math.hypot(mouse_pos[0] - START_POSITION[0], mouse_pos[1] - START_POSITION[1])
    if distance <= CIRCLE_SIZE:
        return True
    return False

def calculate_angle(point1, point2):
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])



# Main game loop
running = True
show_end_position = False
while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Press 'esc' to close the experiment
                running = False
            elif event.key == pygame.K_4: # Press '4' to test pertubation_mode
                pertubation_mode = True
            elif event.key == pygame.K_5: # Press '5' to end pertubation_mode
                pertubation_mode = False
            
    # Design experiment
    if attempts == 1:
        pertubation_mode = False
    elif attempts == 100: #GRAD_START: #40
        pertubation_mode = True
        pertubation_type = 'gradual' 
    elif attempts == 100:  #80
        pertubation_mode = False
    elif attempts == 100:  #120
        pertubation_mode = True    
        pertubation_type = 'sudden'  
    elif attempts == 100:  #80
        pertubation_mode = False
    elif attempts == 5:  #160
        pertubation_mode = False
        shift_target = True
        start_target=math.radians(-30)
    elif attempts == 60:  
        pertubation_mode = False
        start_target=math.radians(START_ANGLE)
    elif attempts == 70:  #160
        pertubation_mode = 'sudden'
    elif attempts == 80:  #160
        pertubation_mode = False
    elif attempts >= ATTEMPTS_LIMIT:
        running = False        

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)
    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Calculate distance from START_POSITION to mouse_pos
    deltax = mouse_pos[0] - START_POSITION[0]
    deltay = mouse_pos[1] - START_POSITION[1]
    distance = math.hypot(deltax, deltay)

    a = attempts-10#GRAD_START
    
    if pertubation_mode:
        # TASK1: CALCULATE perturbed_mouse_pos 
        if pertubation_type == 'gradual':

            t = np.floor(a/5)+1
            angle = -3*t
            print('angle: ',angle)
            

        elif pertubation_type == 'sudden':
            angle = 30
        font = pygame.font.Font(None, 36) 
        score_text = font.render(f"Perturbation angle: {angle}", True, WHITE) 
        screen.blit(score_text, (1000, 200))
        angle = np.deg2rad(angle)
        perturbed_mouse_pos = [
            np.cos(angle)*deltax -np.sin(angle)*deltay + START_POSITION[0],
            np.sin(angle)*deltax + np.cos(angle)*deltay + START_POSITION[1]
                            ]
        circle_pos = perturbed_mouse_pos
    else:
        circle_pos = pygame.mouse.get_pos()
    
    # Check if target is hit or missed
    # hit if circle touches target's center
    if check_target_reached():
        score += 1
        attempts += 1
        new_target = None  # Set target to None to indicate hit
        
        start_time = 0  # Reset start_time after hitting the target

        # CALCULATE AND SAVE ERRORS between target and circle end position for a hit
        error_angle = 0.0
        font = pygame.font.Font(None, 36) 
        score_text = font.render("here:", True, WHITE) 
        screen.blit(score_text, (1000, 200))
        #error_angle = float("NaN")
        if (move_faster):
            error_angle = np.nan
        error_angles.append(error_angle)

    #miss if player leaves the target_radius + 1% tolerance
    elif new_target and math.hypot(circle_pos[0] - START_POSITION[0], circle_pos[1] - START_POSITION[1]) > TARGET_RADIUS*1.01:
        attempts += 1
        new_target = None  # Set target to None to indicate miss
        start_time = 0  # Reset start_time after missing the target


        targetX = WIDTH / 2 
        targetY = HEIGHT / 2 + TARGET_RADIUS

        target_pos = (targetX, targetY)

        error_angle = calculate_angle(START_POSITION, circle_pos) + np.pi/2


        if (shift_target):
            error_angle = error_angle + np.deg2rad(30)
    
    
        if error_angle > np.pi:
           error_angle =  (2*np.pi) - error_angle
        error_angle = np.rad2deg(error_angle)
        error_angle = np.abs(error_angle)
        if (move_faster):
            error_angle = np.nan
        error_angles.append(error_angle)


    # Check if player moved to the center and generate new target
    if not new_target and at_start_position_and_generate_target(mouse_pos):
        new_target = generate_target_position()
        move_faster = False
        start_time = pygame.time.get_ticks()  # Start the timer for the attempt

    # Check if time limit for the attempt is reached
    current_time = pygame.time.get_ticks()
    if start_time != 0 and (current_time - start_time) > TIME_LIMIT:
        move_faster = True
        start_time = 0  # Reset start_time
        
    # Show 'MOVE FASTER!'
    if move_faster:
        font = pygame.font.Font(None, 36)
        text = font.render('MOVE FASTER!', True, RED)
        text_rect = text.get_rect(center=(START_POSITION))
        screen.blit(text, text_rect)

# Generate playing field
    # Draw current target
    if new_target:
        pygame.draw.circle(screen, BLUE, new_target, TARGET_SIZE // 2)

    # Draw circle cursor
    if mask_mode:
        if distance < MASK_RADIUS:
            pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    else:
        pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    
    # Draw start position
    pygame.draw.circle(screen, WHITE, START_POSITION, 5)        

    # Show score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Show attempts
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Attempts: {attempts}", True, WHITE)
    screen.blit(score_text, (10, 30))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
pertubations = [' gradual \n Pertubation', ' no \n  Pertubation', ' sudden \n Pertubation', ' no \n Pertubation']
print(error_angles)

## TASK 2, CALCULATE, PLOT AND SAVE ERRORS from error_angles
error_angles = np.array(error_angles)
att_nr=np.linspace(0,len(error_angles),len(error_angles))
# points are connected between nan values
mask = np.isfinite(error_angles.astype(np.double))
plt.plot(att_nr[mask],error_angles[mask], linestyle = 'dashed')
plt.scatter(att_nr,error_angles)
plt.xlabel('#Attempt')
plt.ylabel('Error Angle (degrees)')
plt.xlim(0,200)
plt.ylim(0, np.nanmax(error_angles+5))
for change in range(len(DESIGN_CHANGE)):
    plt.axvline(x=DESIGN_CHANGE[change], color='red')
    plt.text(DESIGN_CHANGE[change], np.nanmax(error_angles)+4, pertubations[change], color = 'red',rotation=0, va='top')


plt.savefig('reaching_task_graph.png')
plt.show()
sys.exit()