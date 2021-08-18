//******************************************************************************
//
// File:    Bits.java
// Package: edu.rit.numeric
// Unit:    Interface edu.rit.numeric.Bits
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

package edu.rit.numeric;

/**
 * Interface Bits specifies the interface for a group of bit positions to be
 * extracted from a {@linkplain BigInteger}.
 *
 * @see  BigInteger#extractBits(BigInteger,Bits)
 *
 * @author  Alan Kaminsky
 * @version 08-Sep-2016
 */
public interface Bits
	{
	/**
	 * Returns the number of bits to be extracted.
	 *
	 * @return  Number of bits.
	 */
	public int size();

	/**
	 * Returns the <TT>i</TT>-th bit position to be extracted.
	 *
	 * @param  i  Index, 0 &le; <TT>i</TT> &le; <TT>size()</TT>&minus;1.
	 *
	 * @return  <TT>i</TT>-th bit position.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
	 */
	public int bit
		(int i);
	}
