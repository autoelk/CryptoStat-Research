//******************************************************************************
//
// File:    MapperTask.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.MapperTask
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

package edu.rit.pjmr;

import edu.rit.pj2.Section;
import edu.rit.pj2.Task;
import edu.rit.pj2.Vbl;
import edu.rit.util.ActionResult;
import edu.rit.util.Pair;
import java.io.IOException;

/**
 * Class MapperTask provides a mapper task in the Parallel Java Map Reduce
 * Framework. Do not create mapper tasks directly; rather, define mapper tasks
 * as part of a {@linkplain PjmrJob}.
 * <P>
 * Each mapper task in a PJMR job does the following, using the configuration
 * specified in the PJMR job. For further information, refer to the
 * documentation for the classes mentioned below.
 * <OL TYPE=1>
 * <P><LI>
 * For each of the mapper task's configured {@linkplain Source}s, create the
 * source object and call its <TT>open()</TT> method.
 * <P><LI>
 * For each source's configured {@linkplain Mapper}s, create the mapper object.
 * <P><LI>
 * Create the mapper task's configured {@linkplain Combiner}. This is the global
 * combiner.
 * <P><LI>
 * Create the mapper task's configured {@linkplain Customizer}. If no customizer
 * was configured, an instance of the Customizer base class (which does nothing)
 * is created.
 * <P><LI>
 * Call the customizer's <TT>start()</TT> method, passing in any configured
 * argument strings plus the global combiner.
 * <P><LI>
 * Run the mapper(s) in parallel, each in its own thread. Each mapper thread
 * does the following:
 * <OL TYPE=a>
 * <P><LI>
 * Create a thread-local copy of the global combiner.
 * <P><LI>
 * Call the mapper's <TT>start()</TT> method, passing in any configured argument
 * strings plus the thread-local combiner.
 * <P><LI>
 * Repeatedly call the mapper's source's <TT>next()</TT> method to obtain the
 * next data record from the source, and call the mapper's <TT>map()</TT>
 * method, passing in the data record's ID and contents plus the thread-local
 * combiner. (Note that if there is more than one mapper for the same source,
 * all the mappers call the source's <TT>next()</TT> method concurrently; the
 * <TT>next()</TT> method is multiple thread safe.)
 * <P><LI>
 * When there are no more data records, call the mapper's <TT>finish()</TT>
 * method, passing in the thread-local combiner.
 * </OL>
 * <P><LI>
 * After all the mapper threads have finished, reduce the thread-local combiners
 * into the global combiner.
 * <P><LI>
 * Call the customizer's <TT>finish()</TT> method, passing in the global
 * combiner.
 * <P><LI>
 * Call each source's <TT>close()</TT> method.
 * <P><LI>
 * Send the (key, value) pairs from the global combiner to the {@linkplain
 * ReducerTask}.
 * </OL>
 *
 * @param  <I>  Mapper's data record ID data type.
 * @param  <C>  Mapper's data record contents data type.
 * @param  <K>  Mapper's output key data type.
 * @param  <V>  Mapper's output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 16-Jun-2016
 */
public class MapperTask<I,C,K,V extends Vbl>
	extends Task
	{

// Global variables.

	// Source, mapper, and mapper argument strings for each parallel team
	// thread.
	Source<I,C>[] thrSource;
	Mapper<I,C,K,V>[] thrMapper;
	String[][] thrMapperArgs;

	// Combiner.
	Combiner<K,V> combiner;

	// Customizer.
	Customizer<K,V> customizer;

// Task main program.

	/**
	 * Mapper task main program.
	 *
	 * @param  args  Array of command line argument strings.
	 * <UL>
	 * <LI><TT>args[0]</TT> = Node name
	 * </UL>
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Get configuration for this node name.
		String node = args[0];
		MapperTaskConfigTuple<I,C,K,V> configTuple =
			(MapperTaskConfigTuple<I,C,K,V>) takeTuple
				(new MapperTaskConfigTuple<I,C,K,V> (node));
		MapperTaskConfig<I,C,K,V> config = configTuple.config;

		// Set up sources and mappers for each parallel team thread.
		thrSource = (Source<I,C>[]) new Source [config.mapperCount];
		thrMapper = (Mapper<I,C,K,V>[]) new Mapper [config.mapperCount];
		thrMapperArgs = new String [config.mapperCount] [];
		int thr = 0;
		int m = config.sourceConfigList.size();
		for (int i = 0; i < m; ++ i)
			{
			SourceConfig<I,C,K,V> sourceConfig =
				config.sourceConfigList.get (i);
			Source<I,C> source = sourceConfig.source;
			source.open();
			int n = sourceConfig.mapperConfigList.size();
			for (int j = 0; j < n; ++ j)
				{
				MapperConfig<I,C,K,V> mapperConfig =
					sourceConfig.mapperConfigList.get (j);
				thrSource[thr] = source;
				thrMapper[thr] = mapperConfig.newInstance();
				thrMapperArgs[thr] = mapperConfig.mapperArgs;
				++ thr;
				}
			}

		// Set up combiner.
		combiner = configTuple.combinerConfig.newInstance();

		// Set up customizer.
		customizer = config.customizerConfig.newInstance();
		customizer.start (config.customizerConfig.customizerArgs, combiner);

		// Run mappers in parallel.
		parallelDo (config.mapperCount, new Section()
			{
			public void run() throws Exception
				{
				int rank = rank();
				Source<I,C> source = thrSource[rank];
				Mapper<I,C,K,V> mapper = thrMapper[rank];
				Combiner<K,V> thrCombiner = threadLocal (combiner);
				mapper.customizer = customizer;
				mapper.start (thrMapperArgs[rank], thrCombiner);
				Pair<I,C> record;
				while ((record = source.next (rank)) != null)
					mapper.map (record.key(), record.value(), thrCombiner);
				mapper.finish (thrCombiner);
				}
			});

		// Finish customizer.
		customizer.finish (combiner);

		// Close sources.
		for (int i = 0; i < m; ++ i)
			{
			SourceConfig<I,C,K,V> sourceConfig =
				config.sourceConfigList.get (i);
			Source<I,C> source = sourceConfig.source;
			source.close();
			}

		// Send pairs from combiner.
		final PairSender<K,V> pairSender = new PairSender<K,V> (this);
		IOException exc = combiner.forEachItemDo
			(new ActionResult<Pair<K,V>,IOException>()
				{
				IOException exc = null;
				public void run (Pair<K,V> pair)
					{
					try
						{
						if (exc == null)
							pairSender.send (pair);
						}
					catch (IOException exc)
						{
						this.exc = exc;
						}
					}
				public IOException result()
					{
					return exc;
					}
				});
		if (exc != null) throw exc;
		pairSender.close();
		}

	}
