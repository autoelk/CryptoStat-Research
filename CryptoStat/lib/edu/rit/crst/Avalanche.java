//******************************************************************************
//
// File:    Avalanche.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Avalanche
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
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

import edu.rit.gpu.Gpu;
import java.io.IOException;

/**
 * Class Avalanche performs an avalanche {@linkplain Test} for a cryptographic
 * {@linkplain Function}. Each test data series element is the difference
 * (bitwise exclusive-or) between the raw data series element and the
 * <I>first</I> raw data series element.
 *
 * @see Test
 *
 * @author  Alan Kaminsky
 * @version 21-Feb-2018
 */
public class Avalanche
	extends Test
	{

// Exported constructors.

	/**
	 * Construct a new avalanche test object.
	 */
	public Avalanche()
		{
		super();
		}

// Exported operations.

	/**
	 * Get a constructor expression for this test object. The constructor
	 * expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this test object.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return "edu.rit.crst.Avalanche()";
		}

	/**
	 * Get a description of this test.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "Avalanche test";
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/crst/Avalanche.ptx";
		}

	}
