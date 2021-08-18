//******************************************************************************
//
// File:    Complement.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Complement
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.crst;

import edu.rit.util.BigInt;

/**
 * Class Complement provides an object that generates input values for a
 * cryptographic {@linkplain Function Function}.
 * <P>
 * An instance of class Complement is layered on top of another input generator.
 * The latter is specified as an argument to the Complement constructor, in the
 * form of a constructor expression. The Complement constructor uses the
 * constructor expression to create the underlying input generator. (For further
 * information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * Class Complement invokes the underlying input generator to generate a series
 * of cryptographic function input values. Class Complement then complements
 * each input value.
 *
 * @author  Alan Kaminsky
 * @version 15-Nov-2016
 */
public class Complement
	extends TransformGenerator
	{

// Exported constructors.

	/**
	 * Construct a new complement input generator.
	 *
	 * @param  ctor  Constructor expression for the underlying input generator.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>ctor</TT> is null.
	 */
	public Complement
		(String ctor)
		{
		super (ctor);
		}

// Exported operations.

	/**
	 * Get a constructor expression for this input generator. The constructor
	 * expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this input generator.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return "Complement(" + ctor + ")";
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return gen.description() + ", complemented";
		}

// Hidden operations.

	/**
	 * Transform the given input value.
	 *
	 * @param  value  Input value.
	 */
	protected void transform
		(BigInt value)
		{
		value.complement();
		}

	}
