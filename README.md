# puzzle-bot
Install all the requirements and for now run the getsides. It will extract all the sides in the given image. The thresholds and variables are curated for this specific image, so you will need to change it to try on another image. Noitce that you might have to change the binary threshhold to opposite if your puzzle pieces are darker than the background. If your piece sizes and dimensions are different, you will also have to adjust that. Let me explain the pipeline now.

1. Binarize the image
2. Use cv2 to find the contours(the puzle pieces)
3. Extract the points of the piece outline
4. Find discontunites with rdp(ramer douglas pueker) algorithm and find the best match(It took me some time to find the correct epsilon, and i tried other techniques, but this is the simplest)
5. Extract sides from given edges

SIDE CLASSIFICATION

Side type - Find max dev and axis of the side, inputs are side index. output - side type!
Side angle - Angle between first and last point
Adjust angle func - There to normalize side
Calculate length
Calculate Axis
Get mid x & get mid y - Visualization purposes
Get hash for side - just for fun 😎

**Side TYPE classification NOT working properly right now. Please dont raise issues on that!**

Solve.py(WORKING ON IT!!!!!)
match all side combinations with directed_haurordoff and output best possible puzzle solved

**INSPIRATION**
I was getting bored one day and thought of making this, then watched mark robers video where he made his own. This is a much simpler version of that and ive done it over my summer holidays!

PS: Guys im just 11, so like my code *might* not be so organized....😬 Open for suggestions!
