/* A Q-Learning Reinforcement Tetris Agent 
* Copyright (C) 2011, Juan Ignacio Navarro Horniacek
* 
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. 
*/

package tetrisexample;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.util.AgentLoader;


/* Quick reference: 
 * action:
 * 0 - left
 * 1 - right
 * 2 - rotate left (counterclockwise)
 * 3 - rotate right (clockwise)
 * 4 - do nothing
 * 5 - put down
 * 
 *  pieces:
 * 0 - I
 * 1 - O
 * 2 - T
 * 3 - S
 * 4 - Z
 * 5 - L
 * 6 - J
 */

class Qsheet implements Serializable{
	// 666 is the maximum state, 16(4positions*4rotations)*7(tiles) = 112
	int possible_states = 667;   		
	int possible_actions = 112;
	public double values[][] = new double[possible_states][possible_actions];
}

public class QTetrisAgent implements AgentInterface{

	private Action action;
	private int numInts =1;
	private int numDoubles =0;

	boolean firstActionOfEpisode;
	int totalSteps;
	int Steps;
	// The width of the simplified tetris with 175 states
	int TrainingAgentWidth = 4;
	double totalRew;
	int CompletedLines;

	//parameters and valus for the Qlearning algorithm
	double alpha = 0.8;  // in order to have a fast learning
	double gamma = 0.1;
	int current_state;
	int action_taken;
	// 666 is the maximum state, 16(4positions*4rotations)*7(tiles) = 112
	int possible_states = 667;
	int possible_actions = 112;
	Qsheet Qvalues = new Qsheet();
      
	TaskSpec TSO = null;
	
	public static final int  MAXWIDTH = 20;
	public static final int  MAXHEIGHT = 40;
	public static final int PADDING = 3;
	public static final int T_WALL = 8;
	public static final int T_EMPTY = 0;
	int bestrot, bestpos;
	int width, height, piece;
	// [piece][rotation][rows][columns]
	int[][][][] tiles = new int[7][4][4][4];    
	int[][][] tilebottoms = new int[7][4][4];
	// heights of columns
	int[] skyline;

	// in this gameboard I save the state of the game so I don't have to depend in the GUI
	int[][] gameboard;

	static final double MIN_VALUE = -1e10;

	public QTetrisAgent(){
	}

	/**
	* Called just before unloading the current MDP.
	*/
	public void agent_cleanup() {
	}

	/**
	* Called at the end of an episode (that is, when the board is filled up)
	* @param lastrew  the reward received for the last step
	*/
	public void agent_end(double lastrew) {
		totalRew += lastrew;

		try{
			FileOutputStream fos = new  FileOutputStream("Qvalues.dat");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(Qvalues);
			oos.close();
			System.out.printf("The data was saved in the file Qvalues.dat\n");
		}
		catch (IOException e) {
			System.out.printf("The data could not be saved in the file\n");
		}


		System.out.printf("Printing some stats for the Q-Learning method Alpha Version: \t steps:%d \t reward:%.2f \n", totalSteps, totalRew );
		System.out.printf(" \n \n \nQ-Learning method Alpha Version: Completed Lines: %d \n \n \n", CompletedLines);
	}

	/**
	* Is never called.
	*/
	public void agent_freeze() {
	}

	/**
	* Called when starting a new MDP.
	* @param taskSpec  the Task Specification Object of the current MDP.
	*/

	public void agent_init(String taskSpec) {
		TSO = new TaskSpec(taskSpec);
		firstActionOfEpisode = true;

		action = new Action(TSO.getNumDiscreteActionDims(),TSO.getNumDiscreteActionDims());	
		totalRew = 0;
		totalSteps = 0;
		
		try{
			FileInputStream fis = new FileInputStream("Qvalues.dat");
			ObjectInputStream ois = new ObjectInputStream(fis);
			try{
				Qvalues = (Qsheet) ois.readObject();
			}
			catch (ClassNotFoundException e){
				System.out.printf("There was problems with the class qsheet\n");
			}
			ois.close();
			System.out.printf("The data was read from the file Qvalues.dat\n");
		}
		catch (IOException e) {
			System.out.printf("The data could not be read from the file\n");
		}
	}


	public String agent_message(String arg0) {
		return null;
	}
        
	/**
	* Called when a new episode starts. 
	* We perform some initialization: find out the board width&height,
	* set up an empty board and the arrays that contain rotated versions 
	* of the 7 tetrominoes.
	* After initialization, we call agent_step() to generate a suitable action
	* 
	* @param o the first obrervation of the episode
	* @return  the first action of the episode
	*/
	public Action agent_start(Observation o){
		CompletedLines = 0;
		Steps = 0;
		int len = o.intArray.length;
		height = o.intArray[len-2];
		// Here is where I should set it as TrainingAgentWidth in order to train the agent
		width  = o.intArray[len-1];
		// 2*PADDING because I have both top and bottom or right and left
		board = new int[width + 2*PADDING][height + 2*PADDING];	
		workboard = new int[width + 2*PADDING][height + 2*PADDING];
		skyline = new int[width + 2*PADDING];

		// in this gameboard the state of the game is represented
		gameboard = new int[width + 2*PADDING][height + 2*PADDING];

		bestrot = 0;
		bestpos = 0;

		// generate all the tetrominoes and their rotations
		for(int i = 0; i < 7; i++){
			for(int j = 0; j < 4; j++){
				int tile[][] = generateTile(i, j);
				for(int x = 0; x < 4; x++){
					// this is for the tilebottoms
					int last = -100;
					for(int y = 0; y < 4; y++){
						tiles[i][j][x][y] = tile[x][y];
						if(tile[x][y] != 0)
							last = y;
					}
					tilebottoms[i][j][x] = last;		
				}
			}
		}

		// clean the gameboard where I will update the state of the game
		for(int i = 0; i < gameboard.length; i++){
			for(int j = 0; j < gameboard[i].length; j++)
				gameboard[i][j] = T_WALL;
        	}
		for(int i = 0; i < width; i++){
			for(int j = 0; j < height + PADDING; j++)
				gameboard[i + PADDING][j] = T_EMPTY;
		}
		// 4-Do Nothing
		action.intArray[0] = 4;
		return agent_step(0, o);
	}


	/**
	* auxiliary array. When a new tetromino arrives at the board, we detect 
	* its type and the position of its upper-left corner. The positions of 
	* the four "minoes" are stored in this array.
	*/
	int[][][] dpos = {{{0,0},{1,0},{2,0},{3,0}},
				{{0,0},{1,0},{0,1},{1,1}},
				{{0,0},{-1,1},{0,1},{0,2}},
				{{0,0},{1,0},{1,1},{2,1}},
				{{0,0},{1,0},{-1,1},{0,1}},
				{{0,0},{1,0},{2,0},{2,1}},
				{{0,0},{0,1},{-1,1},{-2,1}}
	};
        
	/**
	* auxiliary array. When a new tetromio arrives, we detect its x position 
	* as its upper-left corner. To get the "true position", we have to modify it.
	* PosShift[rot][piece] is the shift required for "piece" in rotation "rot".
	*/
	public static final int[][] PosShift = {
		{0, 0,-1, 0,-1, 0,-2},
		{2, 0,-1, 0,-1, 0,-1},
		{0, 0, 0, 0,-1, 0,-2},
		{0, 0,-1, 0,-1, 1,-2}
	}; 

	// number of different rotations for the tetrominoes.
	public static final int[] nrots = {2,1,4,2,2,4,4}; 
    
	// rawboard contains the currently falling piece
	int[][] rawboard = new int[MAXWIDTH][MAXHEIGHT];

	// board does not contain the currently falling piece, and is padded with zeroes all around
	int[][] board;

	// a work copy of board[][], we try out new placements of tetrominoes here 
	int[][] workboard;

	int pos = 0, rot = 0;
	int pos0 = -1;
        
	/**
	* It selects a primitive action depending on the current game state.
	* Note: the selection logic is in putTileGreedy()
	* 
	* @param lastrew  the last reward
	* @param o     the current game state
	* @return      the action taken by the agent
	*/
	public Action agent_step(double lastrew, Observation o){
		int len = o.intArray.length;
		int i, j, k, a;
		int arrayWidth = o.intArray[len-1];
		boolean isnewpiece;
		action.intArray[0] = 0;
		totalSteps++;
		totalRew += lastrew;
		piece = -1;

		for (i = 0; i < 7; i++){
			if (o.intArray[len-9+i]==1)
				piece = i;
		}
		// I make a copy of the gameboard for my own representation
		for (i = 0; i < arrayWidth; i++){
			for (j = 0; j < height; j++){
				rawboard[i][j] = o.intArray[j*arrayWidth+i];
			}
		}
		isnewpiece = false;
		for (i = 0; i < arrayWidth; i++){
			// if there is something in the first line, then we have a new tetromino
			if (rawboard[i][0]!= 0){
				isnewpiece = true;
				break;
			}
		}
		if (isnewpiece){
			// we overwrite the new piece with a different "color" (2 instead of 1)
			// so that we can separate it from the rest of the board
			// we could also erase it...
			pos0 = i;
			for (k=0; k<4; k++){
				rawboard[pos0+dpos[piece][k][0]][dpos[piece][k][1]] = 2;
			}
		}
		if (firstActionOfEpisode){
			// if it was the first action of the Episode, it isn't anymore
			firstActionOfEpisode = false;
		}

		if (isnewpiece){
			rot = 0;
			pos   = pos0 + PosShift[rot][piece];
			clearBoard();
			for (i=0; i<width; i++){
				for (j=0; j<height; j++){
					// note: the new piece is not copied to board!
					board[i+PADDING][j+PADDING] = (gameboard[i+PADDING][j+PADDING]!=0) ? 1:0;
				}
			}
			updateSkyline();
			// putTileGreedy() analyzes the board and sets bestrot and bestpos
			putTileGreedy(piece);

			/* This is for training the agent
			* int next_state = getState(board);
			* double reward = getValue(board);
			* double thisQ = Qvalues.values[current_state][action_taken];
			* double maxQ = getMaxRewardState(next_state);
			* Qvalues.values[current_state][action_taken] = thisQ + alpha*(reward + gamma*maxQ - thisQ);
			* muestro la matriz de valores
			* ShowQvalues();*/
			// it means that there is a new tile to place
			Steps++;  
			//Print the values such as maxim height in order to use them as metrics
			//int characteristics[] = getCharacteristics(board);
			//System.out.printf("%d %d\n", Steps, characteristics[2]); // amount of holes
		}
		pos = pos0 + PosShift[rot][piece];

		// bestrot and bestpos is set by putTileGreedy, when a new piece arrives.
		// after that, we try to achieve them step-by-step with elementary moves
		// bestrot is a number between 0 and 3, but it may be larger than the 
		// number of _different_ rotations for a given tetromino. 
		// For example, a Z piece can have bestrot=+3, which we modify to +1.
		int nrots = 4;
		if ((piece == 0) || (piece==3) || (piece==4)) nrots = 2; //I, S, Z piece
		if (((piece==0) || (piece==3) || (piece==4)) && (bestrot>=2))
			bestrot-=2;
		if (piece ==1) //O piece
			bestrot = 0;

		// now we have bestpos and bestrot. this is translated to 
		// a sequence of primitive moves as follows:
		// 1. when the piece is in the first line, we make only right/left moves
		// 2. we rotate the piece to its final position
		// 3. we move the piece to its final position
		// 4. we drop it.

		if ((isnewpiece) && (pos > bestpos))
			a = 0; //left
		else if ((isnewpiece) && (pos < bestpos))
			a = 1; //right
		else if ((isnewpiece) && (pos == bestpos))
			a = 4; //do nothing
			// maybe we need to rotate later, but rotation is not always allowed in the first line.
		else if ((rot != bestrot) && ((rot == bestrot+1) || (rot+nrots == bestrot+1)))
			a = 2; //rotate left
		else if ((rot != bestrot) && ((rot == bestrot-1) || (rot-nrots == bestrot-1)))
			a = 3; //rotate right
		else if ((rot != bestrot) && ((rot == bestrot+2) || (rot+nrots == bestrot+2)))
			a = 2; //need to rotate twice, we start by one rotate left
		else if ((rot == bestrot) && (pos > bestpos))
			a = 0; //left
		else if ((rot == bestrot) && (pos < bestpos))
			a = 1; //right
		else if ((rot == bestrot) && (pos == bestpos))
			a = 5; //drop
		else{
			// there is some kind of problem. There should not be any.
			//System.out.printf("%d: (%d,%d) vs (%d,%d) \n", piece, pos,rot,bestpos,bestrot);
			a = 4; 
		}
		action.intArray[0] = a;

		// print debug info
		//System.out.println(debug2DArrayToString(tiles[piece][rot])+piece);
		//System.out.printf("rot:%d, offset:%d\n\n", rot, PosShift[rot][piece]);

		// depending on the action taken, we modify our model of the game state
		//         * 0 - left
		//         * 1 - right
		//         * 2 - rotate left (counterclockwise)
		//         * 3 - rotate right (clockwise)
		//         * 4 - do nothing
		//         * 5 - put down

		switch (action.intArray[0]){
			case 0:
				pos0--;
				break;
			case 1:
				pos0++;
				break;
			case 2:
				rot--;
				if (rot<0)
					rot = 3;
				break;
			case 3:
				rot++;
				if (rot>=4) 
					rot = 0;
				break;
		}
		if (((piece == 0) || (piece==3) || (piece==4)) && (rot>=2))
			rot-=2;
		if (piece==1)
			rot=0;

            return action;
	}
	
	public void clearBoard(){
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++)
				board[i][j] = T_WALL;
        	}
		for(int i = 0; i < width; i++){
			for(int j = 0; j < height + PADDING; j++)
				board[i + PADDING][j] = T_EMPTY;
		}
		updateSkyline();
	}
    
	public void updateSkyline(){
		int i, j;
		for(i=0; i<skyline.length; i++){
			for(j=0; j<board[i].length; j++){
				if (board[i][j] != 0)
				break;
			}
			skyline[i] = j;
		}
	}

	/**
	* Copies board to workboard
	*/
	public void copyWorkBoard(){
		for(int i = PADDING; i < width + PADDING; i++)
			System.arraycopy(board[i], 0, workboard[i], 0, height + 2*PADDING);
	}

	public String debug2DArrayToString(int[][] a){
		int i,j;
		int h = a.length; 
		int w = a[0].length;
		String s = "";
		for (j=0; j<h; j++){
			for (i=0; i<w; i++)
				s = s+a[i][j];
			s = s+"\n";
		}       
		return s;
	}
    
	/**
	* Prints an ASCII representation of the board.
	* useful for debugging.
	*/
	public void debugdrawBoard(int board[][]){
		for(int j = 0; j < height; j++){
			for(int i = 0; i < width; i++)
				System.out.printf("%d", new Object[] {Integer.valueOf(board[i + PADDING][j + PADDING])});
			System.out.println();
		}
		System.out.println("  ");
	}

	public void ShowQvalues(){
		System.out.printf("The Qvalues Matrix\n\n");
		for(int i = 0; i < possible_states; i++){
			for(int j = 0; j < possible_actions; j++){
				System.out.print(Qvalues.values[i][j]);
				System.out.printf(" ");
			}
			System.out.println();
		}
	}

	/**
	* Tries to find a good placement for a tetromino of type "type".
	* assigns values to "bestrot" and "bestpos"
	*  
	* @param type  the type of the tetromino to place. an integer from 0 to 6.
	* @return      the number of lines erased by the proposed placement,
	*              or -1 if the tile cannot be placed.
	*/

	int mistakes;  // amount of times that the tile tries to fit and fails

	public int putTileGreedyTraining(int type){
		double bestvalue = MIN_VALUE;
		double probability = 0;
		Random random = new Random();
		int result = -1;
		int res;

		probability = random.nextDouble();

		if(mistakes == 12){		// If I already tried 12 times to fir a tile, then I will not try anymore
			mistakes = 0;		// set mistakes = 0 and return that I couldn't place it
			return -1;
		}
		if(probability < 0.6 && mistakes < 10){	// Put the tile in a random way, if I tried less than 10 times
			bestrot = (int) (random.nextDouble()*(double)(nrots[type]));
			bestpos = (int) (random.nextDouble()*(double)(width));

			copyWorkBoard();
			res = putTile(workboard, type, bestrot, bestpos);
			if(res >= 0){
				bestvalue = getValue(workboard);
			} 
			else{
				bestvalue = MIN_VALUE;
			}
		}
		else{	// Put the tile in a greedy way
			for(int rot = 0; rot < nrots[type]; rot++){  	// for each rotation
				for(int pos = 0; pos < width; pos++){	// for each position (width should be 4)
					copyWorkBoard();
					res = putTile(workboard, type, rot, pos);
					double value;
					if(res >= 0){
						value = getValue(workboard);
					} 
					else{
						value = MIN_VALUE;
					}
					if(value > bestvalue){
						bestvalue = value;
						bestrot = rot;
						bestpos = pos;
					}
				}
			}
		}
		// if the best placement is legal, then we do it on the real board, too.
		if(bestvalue > MIN_VALUE){
			result = putTile(board, type, bestrot, bestpos);
			// put the tile in my gameboard where I have a representation of the game
			int resaux = putTile(gameboard, type, bestrot, bestpos);
			// determines if it is counting the erased lines or not
			int countIt = -1;
			// erase the completed lines in my own representation of the game
			eraseLines(gameboard, countIt);
			updateSkyline();
			mistakes = 0;
			// say which action have to take place
			action_taken = 	getActionNumber(type, bestrot, bestpos);
		}
		else{ // if it is not a valid action try again
			mistakes++;
			//System.out.printf("Not a valid action, this is my attempt number: %d\n", mistakes);
			putTileGreedy(type);
		}

		return result;
	}


	// ** Here I implement the function that puts the tile using exploration *** //
	//    I take first the gameboard, 
	//		bestvale = MIN_VALUE
	//    		for each sub-gameboard of width 4: (I need a function that returns a state for each subgameboerd given an offset)
	// 			state = obtain the state of the current sub-gamboard
	//			look for the maximum value of i in Qvalues.values[state][i]
	//			action = calculate the action
	//			make a copy of the greater gameboard
	//			try to do that action in the greater gameboard
	//			value = value of the gameboard after performing that action
	//			if (value > best_value)
	//				best_action = action
	//	Out of the loop, now I perform that action in the real gameboard.

	public int putTileGreedy(int type){
		double bestvalue = MIN_VALUE;
		int result = -1;
		int res;
		int action_array[] = new int[3];
		// look between all the posible subgameboards
		for(int i = 0; i < width - TrainingAgentWidth; i++){
			int state = getState(gameboard, i);
			int action_number = getMaxActionState(state);
			action_array = getActionFromNumber(action_number);
			copyWorkBoard();
			res = putTile(workboard, type, action_array[1], (i+action_array[2]));
			double value;
			if(res >= 0){
				value = getValue(workboard);
			} 
			else{
				value = MIN_VALUE;
			}
			if(value > bestvalue){
				bestvalue = value;
				bestrot = action_array[1];
				bestpos = i+action_array[2];
			}
		}

		// if the best placement is legal, then we do it on the real board, too.
		if(bestvalue > MIN_VALUE){
			result = putTile(board, type, bestrot, bestpos);
			// put the tile in my gameboard where I have a representation of the game
			int resaux = putTile(gameboard, type, bestrot, bestpos);
			// determines if it is counting the erased lines or not
			int countIt = -1;
			// erase the completed lines in my own representation of the game
			eraseLines(gameboard, countIt);
			updateSkyline();
		}
		else{ // if it is not a valid action return -1
			result = -1;
		}

		return result;
	}


	/**
	* Puts a tetromino of type "type" on the board "b", wit rotation "rot" and position "pos".
	* @param b     the board to play on. Can be either "board" or "workboard"
	* @param type  type of tetromino to place
	* @param rot   rotation of tetromino
	* @param pos   position of tetromino
	* @return
	*/
	public int putTile(int board[][], int type, int rot, int pos){
		int tile[][] = tiles[type][rot];
		int ofs = 10000;
		int countIt = 1;
		for(int x = 0; x < 4; x++)
			ofs = Math.min(ofs, skyline[x + pos + PADDING] - tilebottoms[type][rot][x] - 1);
		if(ofs < PADDING)
			return -1;
		for(int x = 0; x < 4; x++){
			for(int y = 0; y < 4; y++)
				if(tile[x][y] != 0)
					board[x+pos+PADDING][y+ofs] = type + 1;
		}

		int lastLandingHeight = ofs;
		int result = eraseLines(board, countIt);
		int nholes = 0;
		boolean started = false;
		for(int y = 0; y < height; y++){
			if(board[PADDING][y + PADDING] != 0)
				started = true;
			if(started && board[PADDING][y + PADDING] == 0) 
				nholes++;				
		}

		int startrot = 0;
		int startpos = (width - 1) / 2;
		if(type == 0)
			startpos--;

		int nm = 1 + Math.abs(startpos - pos);
		if(type >= 3 && Math.abs(startrot - rot) == 2)
			nm += 2;
		if(startrot != rot)
			nm++;
		if(lastLandingHeight < nm + 1)
			result = -1;

		return result;
	}

	/**
	* Returns an array containing a tetromino of the required type and rotation.
	* 
	* @param type  Type of tetromino
	* @param rot   Rotation of tetromino
	* @return      array containing the tetromino
	*/
	int [][] generateTile(int type, int rot){
		int [][] t;
		// we copy the basic shape into t
		switch (type){
			case 0: // I
				t = new int[][]{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}; 
				break;
			case 1: // O
				t = new int[][]{{1, 1, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}; 
				break; 
			case 2: // T
				t = new int[][]{{0, 1, 0, 0}, {1, 1, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}; 
				break;                
			case 3: // Z
				t = new int[][]{{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}}; 
				break;                
			case 4: // S
				t = new int[][]{{0, 1, 0, 0}, {1, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}}; 
				break;                
			case 5: // J
				t = new int[][]{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}}; 
				break;                
			case 6: // L
				t = new int[][]{{0, 1, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}}; 
				break;                
			default: // non-existent piece
				t = new int[][]{{1, 1, 1, 1}, {1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}; 
				break; 
		}
		int[][] t2 = new int[4][4];
		// we rotate t to the required position and put it to t2
		int x, y;
		switch (rot){
			case 0:
				for (x=0; x<4; x++)
					for (y=0; y<4; y++)
						t2[x][y] = t[x][y];
						break;
			case 1:
				// 1110      0000     
				// 0100      1000       
				// 0000  ->  1100          
				// 0000      1000          
				for (x=0; x<4; x++)
					for (y=0; y<4; y++)
						t2[x][y] = t[y][3-x];
						break;
			case 2:
				for (x=0; x<4; x++)
					for (y=0; y<4; y++)
						t2[x][y] = t[3-x][3-y];
						break;
			case 3:
				for (x=0; x<4; x++)
					for (y=0; y<4; y++)
						t2[x][y] = t[3-y][x];
						break;                
		}
		int emptyrow = 0;
		int emptycol = 0;

		// determine number of empty columns
		outerloop1:
		for (x=0; x<4; x++){
			for (y=0; y<4; y++){
				if (t2[x][y] != 0)
				break outerloop1;
			}
			emptycol++;
		}
        
		// determine number of empty columns
		outerloop2:
		for (y=0; y<4; y++){
			for (x=0; x<4; x++){
				if (t2[x][y] != 0)
					break outerloop2;
			}
			emptyrow++;
		}

		//we shift t2 so that the array does not begin with an empty row or column
		int[][] t3 = new int[4][4];
        
		for (x=emptycol; x<4; x++){
			for (y=emptyrow; y<4; y++)
				t3[x-emptycol][y-emptyrow] = t2[x][y];
		}
		
		return t3;
	}


	/**
	* Erase complete lines from the board, if there are any
	* @param board the board
	* @return  the number of lines erased
	*/
	int eraseLines(int board[][], int countIt){
		int nErased = 0;
		int y = 0;
		int x = 0;
		boolean isfull = true;

		//debugdrawBoard();
		for(y = height-1; y >= 0; y--){
			// assume that the line is complete
			isfull = true;
			for(x = 0; x < width; x++){	// I go wide in the row
				if(board[x+PADDING][y+PADDING]==0){
				// if I find a hole the row is not complete and I don't need to keep looking
					isfull = false;
					break;	
				}
			}
			if(isfull){
				// if the line is complete I erase it moving all the upper lines one row down
				for(int z = y; z >= 0; z--){
					for(x = 0; x < width; x++){
						board[x+PADDING][z+PADDING] = board[x+PADDING][(z-1)+PADDING];
					}
				}
				nErased++;	// add 1 to the amount of erased lines
				if(countIt == 1){
					CompletedLines++;
				}
				//y++;	// set the index where it was before (because there was one more line before)

			}
		}

		return nErased;
	}

	// The offset cannot be greater than the width of the gameboard - TrainingAgentWidth
	int getState(int board[][], int offset){
		int i, j;
		int stwidth = TrainingAgentWidth;
		int heights[] = new int[stwidth];
		int fixedheights[] = new int[stwidth];
		int differences[] = new int[stwidth-1];
		int result = 0;
		// here I have to read the height of each column, reduce their distances between them in a maxim of 3
		// and calculate the differences in the heights of the first (TrainingAgentWidth) consecutive columns
		// the goal here is that I will give a subgameboard and it will calculate the state.

		// Let's read the heights of the columns:
		for(i = offset; i < (offset + stwidth); i++){
			for(j=0; j<height; j++){
				if (board[i+PADDING][j+PADDING]!= 0) 
				// strat from above and if I find a tile, that's the height
				// to calculate it I use height - j (the empty tiles above the column)
				break;
			}
			heights[i - offset] = height-j;
		}

		// now I have to reduce the distance between them.
		fixedheights[0] = heights[0];
		for(i = 0; i < stwidth-1; i++){
			if(heights[i] - heights[i+1] >= 4) {
				if(fixedheights[i] >= 3) 
					fixedheights[i+1] = fixedheights[i]-3;
				else
					fixedheights[i+1] = 0;
			}
			else if (heights[i] - heights[i+1] <= -4)
				fixedheights[i+1] = fixedheights[i]+3;
			else
				fixedheights[i+1] = fixedheights[i]-(heights[i]-heights[i+1]);
		}

		// now I have to calculate the diference between the heights.
		for(i = 0; i < stwidth-1; i++){
			if(fixedheights[i] - fixedheights[i+1] == 0)
				differences[i] = 0;
			else if (fixedheights[i] - fixedheights[i+1] == -1) // the next column is one tile higher
				differences[i] = 1;
			else if (fixedheights[i] - fixedheights[i+1] == -2) // the next column is two tiles higher
				differences[i] = 2;
			else if (fixedheights[i] - fixedheights[i+1] == -3) // the next column is three tiles higher
				differences[i] = 3;
			else if (fixedheights[i] - fixedheights[i+1] == 1) // the next column is one tile shorter
				differences[i] = 4;
			else if (fixedheights[i] - fixedheights[i+1] == 2) // the next column is two tiles shorter
				differences[i] = 5;
			else if (fixedheights[i] - fixedheights[i+1] == 3) // the next column is three tiles shorter
				differences[i] = 6;
		}


		// now I give each difference as a digit in the state
		int lastdif = stwidth-1;
		for (i = 0; i < lastdif; i++){
			result = result + powerTen(i)*differences[lastdif-1-i]; //start from the last one
		}

		// the result is a number between 000 and 636
		return result;
	}

	int powerTen(int exponent){
		int power = 1;
		for(int i = 0; i < exponent; i++){
			power = power*10;
		}
		return power;
	}

	// return the number of action to perform.
	int getActionNumber(int type, int rot, int pos) {
		int indice = 0;
		int accionesPorTipo = 16;
		int result = 0;

		// for rotation 0
		if(rot == 0 && pos == 0)
			indice = 0;
		if(rot == 0 && pos == 1)
			indice = 1;
		if(rot == 0 && pos == 2)
			indice = 2;
		if(rot == 0 && pos == 3)
			indice = 3;
		// for rotation 1
		if(rot == 1 && pos == 0)
			indice = 4;
		if(rot == 1 && pos == 1)
			indice = 5;
		if(rot == 1 && pos == 2)
			indice = 6;
		if(rot == 1 && pos == 3)
			indice = 7;
		// for rotation 2
		if(rot == 2 && pos == 0)
			indice = 8;
		if(rot == 2 && pos == 1)
			indice = 9;
		if(rot == 2 && pos == 2)
			indice = 10;
		if(rot == 2 && pos == 3)
			indice = 11;
		// for rotation 3
		if(rot == 3 && pos == 0)
			indice = 12;
		if(rot == 3 && pos == 1)
			indice = 13;
		if(rot == 3 && pos == 2)
			indice = 14;
		if(rot == 3 && pos == 3)
			indice = 15;

		result = type*accionesPorTipo + indice;
		
		return result;
	}

	// return the action by the number.
	int[] getActionFromNumber(int number) {		
		int type = 0;
		int rot = 0;
		int pos = 0;
		int indice = 0;
		int accionesPorTipo = 16;
		// three results, type, rotation and position.
		int result[] = new int[3]; 

		indice = number % 16;
		type = indice / 16; // this division gives the number of tile

		// for rotation 0
		if(indice == 0){
			rot = 0; pos = 0;
		}
		else if(indice == 1){
			rot = 0; pos = 1;
		}
		else if(indice == 2){
			rot = 0; pos = 2;
		}
		else if(indice == 3){
			rot = 0; pos = 3;
		}
		// for rotation 1
		else if(indice == 4){
			rot = 1; pos = 0;
		}
		else if(indice == 5){
			rot = 1; pos = 1;
		}
		else if(indice == 6){
			rot = 1; pos = 2;
		}
		else if(indice == 7){
			rot = 1; pos = 3;
		}
		// for rotation 2
		else if(indice == 8){
			rot = 2; pos = 0;
		}
		else if(indice == 9){
			rot = 2; pos = 1;
		}
		else if(indice == 10){
			rot = 2; pos = 2;
		}
		else if(indice == 11){
			rot = 2; pos = 3;
		}
		// for rotation 3
		else if(indice == 12){
			rot = 3; pos = 0;
		}
		else if(indice == 13){
			rot = 3; pos = 1;
		}
		else if(indice == 14){
			rot = 3; pos = 2;
		}
		else if(indice == 15){
			rot = 3; pos = 3;
		}

		result[0] = type;
		result[1] = rot;
		result[2] = pos;
		
		return result;
	}

	// return the max value in the row of Qvalues[next_state]
	double getMaxRewardState(int state){   
		double result = MIN_VALUE;
		
		for(int i = 0; i < possible_actions; i++){
			if(result < Qvalues.values[state][i]){
				result = Qvalues.values[state][i];
			}
		}

		return result;
	}

	int getMaxActionState(int state){
		double value = MIN_VALUE;
		int result = -1;	
		for(int i = 0; i < possible_actions; i++){
			if(value < Qvalues.values[state][i]){
				value = Qvalues.values[state][i];
				result = i;
			}
		}

		return result;
	}

	int[] getCharacteristics(int board[][]){
		int heights[] = new int[width];
		int maxh = 0;
		int nholes = 0;
		int rowsholesArray[] = new int[height];
		int rowsholes = 0;
		int i = 0;
		int j = 0;
		int k = 0;
		// 3 characteristics: maximum height, amount of holes, rows with holes
		int result[] = new int[3];  
		// initialize the array
		for(k=0; k<height; k++){
			rowsholesArray[k] = 0;
		}
		for(i = 0; i < width; i++){
			for(j=0; j<height; j++){
				if (board[i+PADDING][j+PADDING]!= 0) 
				break;
			}
			heights[i] = height-j;

			for(; j < height; j++){
				if(board[i+PADDING][j+PADDING] == 0){
				nholes++;
				// mark the row where I found a hole, I can mark several times.
				rowsholesArray[j] = 1;
				}
			}

			if(heights[i] > maxh)
				maxh = heights[i];
		}

		// Count the amount of rows with holes.
		for(k=0; k<height; k++){
			if (rowsholesArray[k] == 1)
				rowsholes++;
		}
		result[0] = maxh;
		result[1] = nholes;
		result[2] = rowsholes;

		return result;
	}

	/**
	* Assigns a heuristic value to the game state represented by "board"
	* @param board the board
	* @return  a value estimation of "board"
	*/
	double getValue(int board[][]){
		int characteristics[] = getCharacteristics(board);
		int maxh = characteristics[0];
		int nholes = characteristics[1];
		int rowsholes = characteristics[2];		
		
		// a very simple heuristic evaluation function... 
		double value = -maxh - 0.1*nholes - 0.5*rowsholes + 0.1*CompletedLines;

		return value;
	}


		public static void main(String[] args){
			System.out.println("Message from inside the agent: Running the Q-Learning Agent Alpha version!!!");
			AgentLoader L=new AgentLoader(new QTetrisAgent());
			L.run();
		}

}
