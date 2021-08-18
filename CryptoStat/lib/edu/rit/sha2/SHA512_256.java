//******************************************************************************
//
// File:    SHA512_256.java
// Package: edu.rit.sha2
// Unit:    Class edu.rit.sha2.SHA512_256
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
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

package edu.rit.sha2;

import edu.rit.gpu.Gpu;
import edu.rit.util.BigInt;
import java.io.IOException;

/**
 * Class SHA512_256 implements the SHA-512/256 hash function with a 104-byte
 * (832-bit) message. The SHA512_256 cryptographic function's input <I>A</I> is
 * the first portion of the message (512 bits); input <I>B</I> is the second
 * portion of the message (320 bits); output <I>C</I> is the digest (256 bits).
 * <P>
 * Class SHA512_256 computes the digest of one 1024-bit block consisting of the
 * message (832 bits) plus the padding (192 bits). The padding consists of one 1
 * bit, sixty-three 0 bits, and the message length (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 28-Aug-2017
 */
public class SHA512_256
	extends SHA512Base
	{

// Exported constructors.

	/**
	 * Construct a new SHA-512/256 cryptographic function.
	 */
	public SHA512_256()
		{
		super
			(0x22312194FC2BF72CL, 0x9F555FA3C84C64C2L,
			 0x2393B86B6F53B151L, 0x963877195940EABDL,
			 0x96283EE2A88EFFE3L, 0xBE5E1E2553863992L,
			 0x2B0199FC2C85B8AAL, 0x0EB72DDC81C52CA2L);
		}

// Exported operations.

	/**
	 * Get a constructor expression for this cryptographic function. The
	 * constructor expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this cryptographic function.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return "edu.rit.sha2.SHA512_256()";
		}

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "SHA-512/256 hash function";
		}

	/**
	 * Get the bit size of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> bit size.
	 */
	public int C_bitSize()
		{
		return 256;
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/sha2/SHA512_256.ptx";
		}

	/**
	 * Record the current round digest.
	 *
	 * @param  dig  Digest.
	 * @param  a    Working variable a.
	 * @param  b    Working variable b.
	 * @param  c    Working variable c.
	 * @param  d    Working variable d.
	 * @param  e    Working variable e.
	 * @param  f    Working variable f.
	 * @param  g    Working variable g.
	 * @param  h    Working variable h.
	 */
	protected void recordDigest
		(BigInt dig,
		 long a,
		 long b,
		 long c,
		 long d,
		 long e,
		 long f,
		 long g,
		 long h)
		{
		long tmp;
		tmp = a + IV_0;
		dig.value[7] = (int)(tmp >> 32);
		dig.value[6] = (int)(tmp);
		tmp = b + IV_1;
		dig.value[5] = (int)(tmp >> 32);
		dig.value[4] = (int)(tmp);
		tmp = c + IV_2;
		dig.value[3] = (int)(tmp >> 32);
		dig.value[2] = (int)(tmp);
		tmp = d + IV_3;
		dig.value[1] = (int)(tmp >> 32);
		dig.value[0] = (int)(tmp);
		}

	}
