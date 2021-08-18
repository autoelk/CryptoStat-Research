//******************************************************************************
//
// File:    Nola911Block.java
// Package: edu.rit.pjmr.example
// Unit:    Class edu.rit.pjmr.example.Nola911Block
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.pjmr.example;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class Nola911Block encapsulates the geographic coordinates of a square block
 * in the {@linkplain Nola911 Nola911} program.
 *
 * @author  Alan Kaminsky
 * @version 14-Jun-2016
 */
public class Nola911Block
	implements Streamable
	{

// Hidden data members.

	// Scale factor. Each side of the square block is 1/<TT>scale</TT> degrees.
	private static final double scale = 100.0;

	private int lat;
	private int lon;

// Exported constructors.

	/**
	 * Construct a new uninitialized block.
	 */
	private Nola911Block()
		{
		}

	/**
	 * Construct a new block from the given 911 call record.
	 *
	 * @param  record  911 call record.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if a parsing error occurred.
	 * @exception  NumberFormatException
	 *     (unchecked exception) Thrown if a parsing error occurred.
	 */
	public Nola911Block
		(String record)
		{
		int a = record.indexOf ('"', 0);
		if (a == -1) throw new IllegalArgumentException();
		int b = record.indexOf (',', a + 1);
		if (b == -1) throw new IllegalArgumentException();
		int c = record.indexOf (')', b + 2);
		if (c == -1) throw new IllegalArgumentException();
		lat = parse (record.substring (a + 2, b));
		lon = parse (record.substring (b + 2, c));
		}

// Exported operations.

	/**
	 * Returns the latitude of the lower left corner of this block.
	 *
	 * @return  Latitude (units of 0.01 degree).
	 */
	public int lat()
		{
		return lat;
		}

	/**
	 * Returns the longitude of the lower left corner of this block.
	 *
	 * @return  Longitude (units of 0.01 degree).
	 */
	public int lon()
		{
		return lon;
		}

	/**
	 * Returns the latitude of the lower left corner of this block.
	 *
	 * @return  Latitude (degrees).
	 */
	public double latitude()
		{
		return lat/scale;
		}

	/**
	 * Returns the longitude of the lower left corner of this block.
	 *
	 * @return  Longitude (degrees).
	 */
	public double longitude()
		{
		return lon/scale;
		}

	/**
	 * Determine if this block is equal to the given object.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this block equals <TT>obj</TT>, false otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		return
			(obj instanceof Nola911Block) &&
			(this.lat == ((Nola911Block)obj).lat) &&
			(this.lon == ((Nola911Block)obj).lon);
		}

	/**
	 * Returns a hash code for this block.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return (lat << 16) + lon;
		}

	/**
	 * Write this block to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeInt (lat);
		out.writeInt (lon);
		}

	/**
	 * Read this block from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		lat = in.readInt();
		lon = in.readInt();
		}

// Hidden operations.

	/**
	 * Parse the given latitude or longitude string.
	 *
	 * @param  s  Latitude or longitude string (degrees).
	 *
	 * @return  Latitude or longitude, rounded down to nearest
	 *          1/{@link #scale scale} degrees.
	 *
	 * @exception  NumberFormatException
	 *     (unchecked exception) Thrown if a parsing error occurred.
	 */
	private static int parse
		(String s)
		{
		return (int) Math.floor (scale*Double.parseDouble (s));
		}

	}
