//******************************************************************************
//
// File:    TestResult.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.TestResult
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

import edu.rit.gpu.Struct;
import java.nio.ByteBuffer;

/**
 * Class TestResult provides the Java equivalent of the CUDA testResult_t
 * structure.
 *
 * @author  Alan Kaminsky
 * @version 22-Feb-2018
 */
public class TestResult
	extends Struct
	{

// Exported data members.

	/**
	 * Run test log Bayes factor.
	 */
	public double runTestLBF;

	/**
	 * Birthday test log Bayes factor.
	 */
	public double birthdayTestLBF;

	/**
	 * Aggregate log Bayes factor.
	 */
	public double aggLBF;

// Exported constructors.

	/**
	 * Construct a new test result object.
	 */
	public TestResult()
		{
		}

// Exported operations.

	/**
	 * Clear this test result object.
	 */
	public void clear()
		{
		runTestLBF = 0.0;
		birthdayTestLBF = 0.0;
		aggLBF = 0.0;
		}

	/**
	 * Copy the given test result object into this test result object.
	 *
	 * @param  obj  Object to copy.
	 */
	public void copy
		(TestResult obj)
		{
		this.runTestLBF = obj.runTestLBF;
		this.birthdayTestLBF = obj.birthdayTestLBF;
		this.aggLBF = obj.aggLBF;
		}

	/**
	 * Returns the size in bytes of the C struct. The size must include any
	 * internal padding bytes needed to align the fields of the C struct. The
	 * size must include any padding bytes at the end needed to align a series
	 * of C structs in an array.
	 *
	 * @return  Size of C struct (24 bytes).
	 */
	public static long sizeof()
		{
		return 24;
		}

	/**
	 * Write this Java object to the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>toStruct()</TT>
	 * method must write this object's fields into the byte buffer exactly as
	 * the C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to write.
	 */
	public void toStruct
		(ByteBuffer buf)
		{
		buf.putDouble (runTestLBF);
		buf.putDouble (birthdayTestLBF);
		buf.putDouble (aggLBF);
		}

	/**
	 * Read this Java object from the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>fromStruct()</TT>
	 * method must read this object's fields from the byte buffer exactly as the
	 * C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to read.
	 */
	public void fromStruct
		(ByteBuffer buf)
		{
		runTestLBF = buf.getDouble();
		birthdayTestLBF = buf.getDouble();
		aggLBF = buf.getDouble();
		}

	}
