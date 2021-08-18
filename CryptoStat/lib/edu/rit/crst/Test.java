//******************************************************************************
//
// File:    Test.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Test
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
 * Class Test is the abstract base class for an object that performs odds ratio
 * tests for a cryptographic {@linkplain Function Function}. The testing process
 * is as follows:
 * <OL TYPE=1>
 * <P><LI>
 * Take a series of <I>C</I> output values computed by the cryptographic
 * function, dubbed the "raw" data series.
 * <P><LI>
 * From the raw data series, derive a "test" data series. Subclasses of class
 * Test do this derivation in various ways. The test data series is assumed to
 * obey a uniform distribution if the cryptographic function is a random
 * mapping.
 * <P><LI>
 * Apply the "run test" to the test data series, and compute a log Bayes factor.
 * <P><LI>
 * Apply the "birthday test" to the test data series, and compute a log Bayes
 * factor.
 * <P><LI>
 * Compute an aggregate log Bayes factor, which is the sum of the preceding log
 * Bayes factors.
 * </OL>
 * <P>
 * The run test looks at blocks of four consecutive values in the test data
 * series; for each two consecutive values <I>x</I> and <I>y</I> in the block,
 * observes whether <I>x</I>&nbsp;&lt;&nbsp;<I>y</I> or
 * <I>x</I>&nbsp;&ge;&nbsp;<I>y</I>, resulting in a pattern of three comparison
 * outcomes; and hypothesizes that the frequencies of these patterns are as
 * expected for a uniformly distributed test data series.
 * <P>
 * The birthday test looks at blocks of consecutive values in the test data
 * series, determines whether a collision (a repeated value) did or did not
 * occur, and hypothesizes that the frequency of no-collision blocks is as
 * expected for a uniformly distributed test data series.
 * <P>
 * The logarithm of the Bayes factor is given by
 * <P>
 * <CENTER>log (pr(<I>H</I><SUB>1</SUB>|<I>D</I>)/pr(<I>H</I><SUB>2</SUB>|<I>D</I>))</CENTER>
 * <P>
 * where <I>D</I> represents the observed test data series; <I>H</I><SUB>1</SUB>
 * is the hypothesis that data came from a uniform distribution; and
 * <I>H</I><SUB>2</SUB> is the hypothesis that the data did not come from a
 * uniform distribution.
 * <P>
 * If the log Bayes factor is positive, then <I>H</I><SUB>1</SUB>'s probability
 * given the data is greater than <I>H</I><SUB>2</SUB>'s probability given the
 * data; that is, the cryptographic function is more likely to be a random
 * mapping. If the log Bayes factor is negative, the cryptographic function is
 * more likely not to be a random mapping. The larger the absolute value of the
 * log Bayes factor, the greater the likelihood that the function is random or
 * nonrandom.
 * <P>
 * The log Bayes factors are computed for the cryptographic function reduced to
 * 1, 2, .&nbsp;.&nbsp;. <I>R</I> rounds, where <I>R</I> is the full number of
 * rounds for the function. Typically, a cryptographic function exhibits
 * nonrandom behavior for small numbers of rounds, but starts to exhibit random
 * behavior after a certain number of rounds. The number of rounds exhibiting
 * random behavior, divided by the total number of rounds, is the function's
 * "randomness margin." A large randomness margin is desirable.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2018
 */
public abstract class Test
	extends Computation
	{

// Exported constructors.

	/**
	 * Construct a new test object.
	 */
	public Test()
		{
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
	public abstract String constructor();

	/**
	 * Get a description of this test object.
	 *
	 * @return  Description.
	 */
	public abstract String description();

	/**
	 * Get the GPU kernel for this test object.
	 *
	 * @param  gpu  GPU accelerator.
	 *
	 * @return  Kernel.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public TestKernel kernel
		(Gpu gpu)
		throws IOException
		{
		return kernel (gpu, TestKernel.class);
		}

	}
