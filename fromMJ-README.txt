EXPLANATION OF GESTURES: 
basically, we mainly look for static gestures, and for certain static gestures, we enter states to do those actions. 
GO is dependent on STOP being the gesture before. 
difference between STOP and HOLD, is that STOP stops the movement of the drone indefinitely until released. HOLD keeps the drone stationary until there is a new gesture or there is no more landmark in frame. 
dynamic gestures i haven't finished but the bare bones of the kNN model is there 


TODO: 
- add finger extension function that recognises whether fingers are folded or not 
- change logic for accelerate/decelerate 
- retrain based on different gestures for translate and rotate (they're too similar when entering action state)

