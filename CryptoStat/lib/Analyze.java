//******************************************************************************
//
// File:    Analyze.java
// Unit:    Class Analyze
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

import edu.rit.crst.BitGroup;
import edu.rit.crst.Function;
import edu.rit.crst.FunctionKernel;
import edu.rit.crst.Generator;
import edu.rit.crst.Test;
import edu.rit.crst.TestKernel;
import edu.rit.crst.TestResult;
import edu.rit.crst.Version;
import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuIntMatrix;
import edu.rit.gpu.GpuStructMatrix;
import edu.rit.pj2.Task;
import edu.rit.util.GpuBigIntArray;
import edu.rit.util.GpuBigIntArray3D;
import edu.rit.util.Instance;
import java.util.Date;

/**
 * Class Analyze is a GPU accelerated parallel program that analyzes a
 * cryptographic function by performing odds ratio tests.
 * <P>
 * Usage: <TT>java pj2 Analyze [-v] [-nt=<I>NT</I>] "<I>function</I>"
 * "<I>generator_A</I>" "<I>generator_B</I>" "<I>test</I>"
 * "<I>bitGroup</I>"</TT>
 * <P>
 * The <I>function</I> argument is a constructor expression for the {@linkplain
 * Function Function} object that implements the cryptographic function to be
 * analyzed. (For further information about constructor expressions, see class
 * {@linkplain edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * The <I>generator_A</I> argument is a constructor expression for the
 * {@linkplain Generator Generator} object that generates a series of A input
 * values for the cryptographic function.
 * <P>
 * The <I>generator_B</I> argument is a constructor expression for the
 * {@linkplain Generator Generator} object that generates a series of B input
 * values for the cryptographic function.
 * <P>
 * The <I>test</I> argument is a constructor expression for the {@linkplain Test
 * Test} object that computes a series of odds ratios from the outputs of the
 * cryptographic function.
 * <P>
 * The <I>bitGroup</I> argument is a constructor expression for the {@linkplain
 * BitGroup BitGroup} object that specifies a series of output bit groups to be
 * tested.
 * <P>
 * The Analyze program analyzes the cryptographic function specified by the
 * <I>function</I> argument, as follows:
 * <OL TYPE=1>
 * <P><LI>
 * Generate a series of <I>A</I> inputs for the cryptographic function as
 * specified by the <I>generator_A</I> argument.
 * <P><LI>
 * Generate a series of <I>B</I> inputs for the cryptographic function as
 * specified by the <I>generator_B</I> argument.
 * <P><LI>
 * For each <I>A</I> input, each <I>B</I> input, and each round of the
 * cryptographic function, compute the function's <I>C</I> output.
 * <P><LI>
 * Perform the test specified by the <I>test</I> object on the series of bit
 * groups in the <I>C</I> outputs as specified by the <I>bitGroup</I> object.
 * Each test calculates the log Bayes factor for the hypothesis that the bit
 * group is random. (See class {@linkplain Test Test} for a more precise
 * explanation.) For each round <I>r</I> of the cryptographic function and each
 * bit group <I>g</I>:
 * <OL TYPE=a>
 * <P><LI>
 * For each <I>A</I> input value <I>a</I>, iterate over all the <I>B</I> input
 * values, examine the <I>C</I> output values, and compute the log Bayes factor
 * <I>LBF1</I><SUB><I>r,g,a</I></SUB>. However, if there is only one <I>B</I>
 * input value, this step is omitted, and <I>LBF1</I><SUB><I>r,g,a</I></SUB> is
 * set to zero.
 * <P><LI>
 * For each <I>B</I> input value <I>b</I>, iterate over all the <I>A</I> input
 * values, examine the <I>C</I> output values, and compute the log Bayes factor
 * <I>LBF2</I><SUB><I>r,g,b</I></SUB>. However, if there is only one <I>A</I>
 * input value, this step is omitted, and <I>LBF2</I><SUB><I>r,g,b</I></SUB> is
 * set to zero.
 * </OL>
 * </OL>
 * <P>
 * The Analyze program prints its results, as follows:
 * <OL TYPE=1>
 * <P><LI>
 * For each <I>A</I> input value <I>a</I> and each round <I>r</I>, print the log
 * Bayes factor aggregated over all bit groups,
 * &Sigma;<SUB><I>g</I></SUB>&nbsp;<I>LBF1</I><SUB><I>r,g,a</I></SUB>.
 * (Log Bayes factors are aggregated by adding them together.) This yields
 * insight into the cryptographic function's randomness for specific <I>A</I>
 * input values. However, if there is only one <I>B</I> input value, this
 * printout is omitted.
 * <P><LI>
 * For each <I>B</I> input value <I>b</I> and each round <I>r</I>,
 * print the log Bayes factor aggregated over all bit groups,
 * &Sigma;<SUB><I>g</I></SUB>&nbsp;<I>LBF2</I><SUB><I>r,g,b</I></SUB>.
 * This yields insight into the cryptographic function's randomness for specific
 * <I>B</I> input values. However, if there is only one <I>A</I> input value,
 * this printout is omitted.
 * <P><LI>
 * For each bit group <I>g</I> and each round <I>r</I>, print the log Bayes
 * factor aggregated over all <I>A</I> and <I>B</I> input values,
 * &Sigma;<SUB><I>a</I></SUB>&nbsp;<I>LBF1</I><SUB><I>r,g,a</I></SUB>&nbsp;+&nbsp;&Sigma;<SUB><I>b</I></SUB>&nbsp;<I>LBF2</I><SUB><I>r,g,b</I></SUB>.
 * This yields insight into the cryptographic function's randomness for specific
 * bit groups.
 * <P><LI>
 * For each round <I>r</I>, print the log Bayes
 * factor aggregated over all <I>A</I> and <I>B</I> input values and all bit groups,
 * &Sigma;<SUB><I>g</I></SUB>&nbsp;(&Sigma;<SUB><I>a</I></SUB>&nbsp;<I>LBF1</I><SUB><I>r,g,a</I></SUB>&nbsp;+&nbsp;&Sigma;<SUB><I>b</I></SUB>&nbsp;<I>LBF2</I><SUB><I>r,g,b</I></SUB>).
 * This yields insight into the cryptographic function's overall randomness.
 * <P><LI>
 * Print the number of nonrandom rounds. This is the highest round whose
 * aggregate log Bayes factor was negative in the preceding printout of overall
 * results.
 * <P><LI>
 * Print the randomness margin =
 * 1&nbsp;&minus;&nbsp;(nonrandom&nbsp;rounds)/(total&nbsp;rounds).
 * </OL>
 * <P>
 * The <TT>-v</TT> flag turns on verbose output. In addition to the above, this
 * includes the cryptographic function's series of <I>A</I> and <I>B</I> input
 * values, the cryptographic function's <I>C</I> output values, and the log
 * Bayes factors for each round, bit group, and input value series.
 * <P>
 * If the <TT>-nt=<I>NT</I></TT> flag is specified, <I>NT</I> gives the number
 * of GPU threads per block. If not specified, <I>NT</I> = 768 is used.
 *
 * @author  Alan Kaminsky
 * @version 21-Feb-2018
 */
public class Analyze
	extends Task
	{

	// Command line arguments.
	boolean verbose;
	String functionSpec;
	String generator_A_Spec;
	String generator_B_Spec;
	String testSpec;
	String bitGroupSpec;

	// GPU accelerator.
	Gpu gpu;
	int numBlocks;
	int numThreads = 768;

	// Cryptographic function object.
	Function func;

	// Cryptographic function inputs and outputs.
	int NA;
	int Asize;
	GpuBigIntArray A;
	int NB;
	int Bsize;
	GpuBigIntArray B;
	int R;
	int Csize;
	GpuBigIntArray3D C;

	// Test object, bit group object, and test results.
	Test test;
	BitGroup bitGroup;
	int NBG;
	int NINPUT;
	TestResult[][][] testResult;
	GpuStructMatrix<TestResult> devTestResult;
	GpuIntMatrix pos;

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		long t1 = System.currentTimeMillis();

		// Parse command line arguments.
		int argi = 0;
		while (argi < args.length && args[argi].charAt(0) == '-')
			{
			if (args[argi].equals ("-v"))
				verbose = true;
			else if (args[argi].startsWith ("-nt="))
				{
				try
					{
					numThreads = Integer.parseInt (args[argi].substring (4));
					if (numThreads < 1)
						throw new NumberFormatException();
					}
				catch (NumberFormatException exc)
					{
					error (args[argi] + " illegal");
					}
				}
			else
				usage();
			++ argi;
			}
		if (args.length - argi != 5) usage();
		functionSpec = args[argi];
		generator_A_Spec = args[argi+1];
		generator_B_Spec = args[argi+2];
		testSpec = args[argi+3];
		bitGroupSpec = args[argi+4];

		// Print provenance.
		System.out.printf ("$ java pj2 Analyze");
		if (verbose) System.out.printf (" -v");
		System.out.printf (" -nt=%d", numThreads);
		System.out.printf (" \"%s\"", functionSpec);
		System.out.printf (" \"%s\"", generator_A_Spec);
		System.out.printf (" \"%s\"", generator_B_Spec);
		System.out.printf (" \"%s\"", testSpec);
		System.out.printf (" \"%s\"", bitGroupSpec);
		System.out.println();
		System.out.printf ("CryptoStat v%d%n", Version.CRYPTOSTAT_VERSION);
		System.out.printf ("Started %s%n", new Date (t1));
		System.out.printf ("Directory %s%n", System.getProperty ("user.dir"));
		System.out.printf ("CLASSPATH %s%n", System.getenv ("CLASSPATH"));
		System.out.flush();

		// Set up GPU accelerator.
		gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);
		numBlocks = gpu.getMultiprocessorCount();
		System.out.printf ("%s%n", gpu);
		System.out.printf ("%d blocks, %d threads per block%n",
			numBlocks, numThreads);
		System.out.flush();

		// Create cryptographic function object.
		func = (Function) Instance.newInstance (functionSpec);
		R = func.rounds();
		System.out.printf ("Function = %s%n", func.description());
		System.out.printf ("  A input = %s (%d bits)%n",
			func.A_description(), func.A_bitSize());
		System.out.printf ("  B input = %s (%d bits)%n",
			func.B_description(), func.B_bitSize());
		System.out.printf ("  C output = %s (%d bits)%n",
			func.C_description(), func.C_bitSize());
		System.out.printf ("  Rounds = %d%n", R);
		System.out.flush();

		// Create A input values, upload to GPU.
		Generator gen_A = (Generator) Instance.newInstance (generator_A_Spec);
		gen_A.setFunction (func, true);
		A = gen_A.generate();
		NA = A.item.length;
		Asize = A.wordSize();
		System.out.printf ("A input generator = %s%n", gen_A.description());
		System.out.printf ("  Number of inputs = %d%n", NA);
		System.out.printf ("  Initial value = %s%n", A.item[0]);
		System.out.flush();
		A.allocate (gpu);
		A.hostToDev();

		// Create B input values, upload to GPU.
		Generator gen_B = (Generator) Instance.newInstance (generator_B_Spec);
		gen_B.setFunction (func, false);
		B = gen_B.generate();
		NB = B.item.length;
		Bsize = B.wordSize();
		System.out.printf ("B input generator = %s%n", gen_B.description());
		System.out.printf ("  Number of inputs = %d%n", NB);
		System.out.printf ("  Initial value = %s%n", B.item[0]);
		System.out.flush();
		B.allocate (gpu);
		B.hostToDev();

		// Sanity check on input series lengths.
		if (NA <= 1 && NB <= 1)
			error (String.format ("NA = %d and NB = %d illegal", NA, NB));

		// Create C output values, upload to GPU.
		C = new GpuBigIntArray3D (func.C_bitSize(), NA, NB, R);
		Csize = C.wordSize();
		C.allocate (gpu);
		C.hostToDev();

		// Create test and bit group objects.
		test = (Test) Instance.newInstance (testSpec);
		bitGroup = (BitGroup) Instance.newInstance (bitGroupSpec);
		bitGroup.setFunction (func);
		NBG = bitGroup.count();
		System.out.printf ("Odds ratio test = %s%n", test.description());
		System.out.printf ("%s%n", bitGroup.description());
		System.out.printf ("  Number of bit groups = %d%n", NBG);
		System.out.flush();
		System.out.printf ("  #\tBit Group Description%n");
		for (int g = 0; g < NBG; ++ g)
			{
			System.out.printf ("  %d\t%s%n",
				g, bitGroup.bitGroupDescription (g));
			System.out.flush();
			}

		// Verbose print function inputs.
		if (verbose)
			{
			System.out.printf ("%n======== A INPUTS ========%n");
			System.out.printf ("#\tA Input Value%n");
			System.out.flush();
			for (int i = 0; i < NA; ++ i)
				{
				System.out.printf ("%d\t%s%n", i, A.item[i]);
				System.out.flush();
				}
			System.out.printf ("%n======== B INPUTS ========%n");
			System.out.printf ("#\tB Input Value%n");
			System.out.flush();
			for (int i = 0; i < NB; ++ i)
				{
				System.out.printf ("%d\t%s%n", i, B.item[i]);
				System.out.flush();
				}
			}

		// Evaluate cryptographic function in parallel.
		FunctionKernel kernel = func.functionKernel (gpu);
		kernel.setGridDim (numBlocks);
		kernel.setBlockDim (numThreads);
		kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);
		kernel.evaluateFunction
			(NA, Asize, A.array(),
			 NB, Bsize, B.array(),
			 R,  Csize, C.array());

		// Verbose print function outputs.
		if (verbose)
			{
			C.devToHost();
			System.out.printf ("%n======== C OUTPUTS ========%n");
			System.out.printf ("A#\tB#\tRound\tC Output Value%n");
			System.out.flush();
			for (int i = 0; i < NA; ++ i)
				for (int j = 0; j < NB; ++ j)
					for (int r = 0; r < R; ++ r)
						{
						System.out.printf ("%d\t%d\t%d\t%s%n",
							i, j, r, C.item[i][j][r]);
						System.out.flush();
						}
			}

		// Set up odds ratio test kernel.
		NINPUT = NA + NB;
		testResult = new TestResult [NINPUT] [R] [NBG];
		for (int i = 0; i < NINPUT; ++ i)
			for (int j = 0; j < R; ++ j)
				for (int k = 0; k < NBG; ++ k)
					testResult[i][j][k] = new TestResult();
		devTestResult = gpu.getStructMatrix (TestResult.class, R, NBG);
		for (int j = 0; j < R; ++ j)
			for (int k = 0; k < NBG; ++ k)
				devTestResult.item[j][k] = new TestResult();
		pos = gpu.getIntMatrix (bitGroup.count(), bitGroup.bitSize());
		bitGroup.bitGroups (pos.item);
		pos.hostToDev();
		TestKernel tkernel = test.kernel (gpu);
		tkernel.setGridDim (numBlocks);
		tkernel.setBlockDim (numThreads);
		tkernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);

		// Compute odds ratios for each A input and all B inputs.
		if (NB > 1)
			for (int a = 0; a < NA; ++ a)
				{
				tkernel.computeTest
					(NA, NB, R, Csize, C.array(), NBG, devTestResult,
					 bitGroup.bitSize(), pos, a, -1);
				recordLbf (testResult[a]);
				}

		// Compute odds ratios for each B input and all A inputs.
		if (NA > 1)
			for (int b = 0; b < NB; ++ b)
				{
				tkernel.computeTest
					(NA, NB, R, Csize, C.array(), NBG, devTestResult,
					 bitGroup.bitSize(), pos, -1, b);
				recordLbf (testResult[NA+b]);
				}

		// Verbose print odds ratios.
		if (verbose)
			{
			System.out.printf ("%n======== DETAILED ODDS RATIOS ========%n");
			for (int i = 0; i < NINPUT; ++ i)
				for (int g = 0; g < NBG; ++ g)
					{
					if (i < NA && NB > 1)
						System.out.printf ("A input %d, all B inputs", i);
					else if (i >= NA && NA > 1)
						System.out.printf ("All A inputs, B input %d", i - NA);
					else
						continue;
					System.out.printf (", bit group %d = %s%n",
						g, bitGroup.bitGroupDescription (g));
					System.out.printf
						("Round  Run LBF     Bday LBF    Agg LBF%n");
					System.out.flush();
					for (int r = 0; r < R; ++ r)
						{
						System.out.printf ("%-7d%-12.5g%-12.5g%-12.5g%n",
							r,
							testResult[i][r][g].runTestLBF,
							testResult[i][r][g].birthdayTestLBF,
							testResult[i][r][g].aggLBF);
						System.out.flush();
						}
					}
			}

		// Compute odds ratios aggregated over all bit groups.
		double[][] agglbf_1 = new double [NINPUT] [R];
		for (int i = 0; i < NINPUT; ++ i)
			for (int r = 0; r < R; ++ r)
				{
				double sumlbf = 0.0;
				for (int g = 0; g < NBG; ++ g)
					sumlbf += testResult[i][r][g].aggLBF;
				agglbf_1[i][r] = sumlbf;
				}

		// Print odds ratios for A inputs.
		if (NB > 1)
			{
			System.out.printf ("%n======== ODDS RATIOS, ONE A INPUT, ALL B INPUTS ========%n");
			System.out.printf ("Log Bayes factors aggregated over all bit groups%n");
			printRoundLabels (R);
			for (int i = 0; i < NA; ++ i)
				printAgglbf (R, String.format ("A[%d]", i), agglbf_1[i]);
			}

		// Print odds ratios for B inputs.
		if (NA > 1)
			{
			System.out.printf ("%n======== ODDS RATIOS, ALL A INPUTS, ONE B INPUT ========%n");
			System.out.printf ("Log Bayes factors aggregated over all bit groups%n");
			printRoundLabels (R);
			for (int i = NA; i < NINPUT; ++ i)
				printAgglbf (R, String.format ("B[%d]", i - NA), agglbf_1[i]);
			}

		// Compute odds ratios aggregated over all A and B inputs.
		double[][] agglbf_2 = new double [NBG] [R];
		for (int g = 0; g < NBG; ++ g)
			for (int r = 0; r < R; ++ r)
				{
				double sumlbf = 0.0;
				for (int i = 0; i < NINPUT; ++ i)
					sumlbf += testResult[i][r][g].aggLBF;
				agglbf_2[g][r] = sumlbf;
				}

		// Print odds ratios for bit groups.
		System.out.printf ("%n======== ODDS RATIOS, C OUTPUT BIT GROUPS ========%n");
		System.out.printf ("Log Bayes factors aggregated over all A and B inputs%n");
		printRoundLabels (R);
		for (int g = 0; g < NBG; ++ g)
			printAgglbf (R, String.format ("BG[%d]", g), agglbf_2[g]);

		// Compute odds ratios aggregated over all A and B inputs and all bit
		// groups. Also find largest round exhibiting nonrandom behavior.
		double[] agglbf_3 = new double [R];
		int maxnr = -1;
		for (int r = 0; r < R; ++ r)
			{
			double sumlbf = 0.0;
			for (int g = 0; g < NBG; ++ g)
				sumlbf += agglbf_2[g][r];
			agglbf_3[r] = sumlbf;
			if (sumlbf < 0.0)
				maxnr = Math.max (maxnr, r);
			}

		// Print overall odds ratios.
		System.out.printf ("%n======== ODDS RATIOS, OVERALL ========%n");
		System.out.printf ("Log Bayes factors aggregated over all A and B inputs and all bit groups%n");
		printRoundLabels (R);
		printAgglbf (R, "All", agglbf_3);

		// Print number of rounds exhibiting nonrandom behavior.
		System.out.printf ("%nNonrandom rounds = %d%n", maxnr + 1);
		System.out.printf ("Randomness margin = %.5f%n",
			1.0 - (double)(maxnr + 1)/(double)R);

		long t2 = System.currentTimeMillis();
		System.out.printf ("%nFinished %s%n", new Date (t2));
		System.out.printf ("%d msec%n", t2 - t1);
		}

	/**
	 * Download test results from GPU and record in given matrix.
	 */
	private void recordLbf
		(TestResult[][] tr)
		{
		devTestResult.devToHost();
		for (int i = 0; i < R; ++ i)
			for (int j = 0; j < NBG; ++ j)
				tr[i][j].copy (devTestResult.item[i][j]);
		}

	/**
	 * Print round labels.
	 */
	private static void printRoundLabels
		(int R)
		{
		System.out.printf ("Inputs      ");
		for (int r = 0; r < R; ++ r)
			System.out.printf ("%-12s", String.format ("Round %d", r));
		System.out.println();
		System.out.flush();
		}

	/**
	 * Print aggregate log Bayes factors.
	 */
	private static void printAgglbf
		(int R,
		 String label,
		 double[] agglbf)
		{
		System.out.printf ("%-12s", label);
		for (int r = 0; r < R; ++ r)
			System.out.printf ("%-12.5g", agglbf[r]);
		System.out.println();
		System.out.flush();
		}

	/**
	 * Print an error message and exit.
	 */
	private static void error
		(String msg)
		{
		System.err.printf ("Analyze: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 Analyze [-v] [-nt=<NT>] \"<function>\" \"<generator_A>\" \"<generator_B>\" \"<test>\" \"<bitGroup>\"");
		System.err.println ("-v = Print verbose output");
		System.err.println ("<NT> = Number of GPU threads per block (default 768)");
		System.err.println ("<function> = Cryptographic function constructor expression");
		System.err.println ("<generator_A> = A input generator constructor expression");
		System.err.println ("<generator_B> = B input generator constructor expression");
		System.err.println ("<test> = Odds ratio test object constructor expression");
		System.err.println ("<bitGroup> = Bit group object constructor expression");
		terminate (1);
		}

	/**
	 * This program requires one CPU core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	/**
	 * This program requires one GPU accelerator.
	 */
	protected static int gpusRequired()
		{
		return 1;
		}

	}
