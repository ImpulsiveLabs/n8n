import type {
	INodeType,
	INodeTypeDescription,
	INodeExecutionData,
	IExecuteFunctions,
	IHttpRequestOptions,
	IHttpRequestMethods,
} from 'n8n-workflow';
import { ApplicationError, NodeConnectionType } from 'n8n-workflow';
import type { Iso3166Alpha2CountryCode, WorkResponse } from 'openalex-ts';

interface TopicCluster {
	id: string;
	display_name: string;
	count: number;
}

interface ClusterMapEntry {
	topic_id: string;
	work_name: string;
	countries: string[];
	cluster_name: string;
	created_date: string;
	authors: string[];
	institutions: Array<{ display_name: string; country_code: Iso3166Alpha2CountryCode }>;
}

interface CalendarEntry {
	date: string;
	value: number;
	category: string;
	properties: { article_name: string };
}

interface AuthorNode {
	name: string;
	category: string;
}

interface AuthorLink {
	source: string;
	target: string;
	value: number;
}

interface InstitutionNode {
	name: string;
	category: string;
}

interface InstitutionLink {
	source: string;
	target: string;
	value: number;
}

interface Concept {
	name: string;
	value: number;
}

interface Keyword {
	name: string;
	value: number;
}

interface Article {
	id: string;
	display_name: string;
	cited_by_count: number;
	publication_year: number;
	counts_by_year: Array<{ year: number; cited_by_count: number }>;
}

interface CitationsPerYear {
	headers: string[];
	values: Record<string, number[]>;
}

interface VisualizationResult {
	calendar_publications: { calendar: CalendarEntry[] };
	'pie-chart_countries': { values: Record<string, number> };
	'hierarchical-edge-bundling_co-authorship': { nodes: AuthorNode[]; links: AuthorLink[] };
	'force-directed-graph_co-authorship': { nodes: AuthorNode[]; links: AuthorLink[] };
	'hierarchical-edge-bundling_institutions': { nodes: InstitutionNode[]; links: InstitutionLink[] };
	'force-directed-graph_institutions': { nodes: InstitutionNode[]; links: InstitutionLink[] };
	'word-cloud_concepts': { words: Concept[] };
	'word-cloud_keywords': { words: Keyword[] };
	'bar-chart_citations': CitationsPerYear;
}

function extractTopPrimaryTopicClusters(works: WorkResponse[]): TopicCluster[] {
	const topicCounts: Record<string, TopicCluster> = {};

	works.forEach(work => {
		if (work.primary_topic) {
			const topic = work.primary_topic;
			const key = topic.id;
			if (!topicCounts[key]) {
				topicCounts[key] = {
					id: key,
					display_name: topic.display_name,
					count: 0,
				};
			}
			topicCounts[key].count++;
		}
	});

	return Object.values(topicCounts)
		.sort((a, b) => b.count - a.count)
		.slice(0, 6);
}

async function createVisualizations(works: WorkResponse[], topClusters: TopicCluster[]): Promise<VisualizationResult> {
	const nameCounts: Record<string, number> = {};
	works.forEach(work => {
		const name = work.display_name;
		if (name) {
			nameCounts[name] = (nameCounts[name] || 0) + 1;
		}
	});

	const clusterMap: Record<string, ClusterMapEntry> = {};
	works.forEach(work => {
		if (work.primary_topic && work.display_name && work.created_date && work.authorships) {
			const topicId = work.primary_topic.id;
			const name = work.display_name;
			if (topClusters.some(cluster => cluster.id === topicId)) {
				const countries = [...new Set(
					work.authorships
						.flatMap(auth => auth.countries)
						.filter((country): country is Iso3166Alpha2CountryCode => !!country)
				)];

				clusterMap[work.id] = {
					topic_id: topicId,
					work_name: name,
					countries,
					cluster_name: work?.primary_topic.display_name,
					created_date: work?.created_date,
					authors: work?.authorships
						.map(auth => auth.author?.display_name)
						// eslint-disable-next-line @typescript-eslint/no-shadow
						.filter((name): name is string => !!name),
					institutions: work.authorships
						.flatMap(auth =>
							(auth.institutions || [])
								.map(inst => ({
									display_name: inst.display_name,
									country_code: inst.country_code || (auth.countries && auth.countries[0]),
								}))
								.filter((inst): inst is { display_name: string; country_code: Iso3166Alpha2CountryCode } => !!inst.display_name && !!inst.country_code)
						),
				};
			}
		}
	});

	const calendar: CalendarEntry[] = Object.keys(clusterMap)
		.filter(workId => topClusters.some(cluster => cluster.id === clusterMap[workId].topic_id) && clusterMap[workId].created_date !== undefined && clusterMap[workId].created_date !== null)
		.map(workId => ({
			date: clusterMap[workId].created_date,
			value: 1,
			category: clusterMap[workId].cluster_name,
			properties: {
				article_name: clusterMap[workId].work_name,
			},
		}));

	const countryCounts: Record<string, number> = {};
	Object.values(clusterMap).forEach(work => {
		work.countries.forEach(country => {
			countryCounts[country] = (countryCounts[country] || 0) + 1;
		});
	});

	const topCountryCounts = Object.fromEntries(
		Object.entries(countryCounts)
			.sort((a, b) => b[1] - a[1])
			.slice(0, 6)
	);

	const authorCounts: Record<string, number> = {};
	const authorCategories: Record<string, string> = {};
	const coauthorshipCounts: Record<string, number> = {};

	Object.values(clusterMap).forEach(work => {
		work.authors.forEach(author => {
			authorCounts[author] = (authorCounts[author] || 0) + 1;
			if (!authorCategories[author]) {
				authorCategories[author] = work.cluster_name;
			}
		});

		for (let i = 0; i < work.authors.length; i++) {
			for (let j = i + 1; j < work.authors.length; j++) {
				const pair = [work.authors[i], work.authors[j]].sort().join(',');
				coauthorshipCounts[pair] = (coauthorshipCounts[pair] || 0) + 1;
			}
		}
	});

	const authorNodes: AuthorNode[] = Object.keys(authorCounts)
		.filter(author => authorCounts[author] >= 2)
		.map(author => ({
			name: author,
			category: authorCategories[author],
		}));

	const authorLinks: AuthorLink[] = Object.keys(coauthorshipCounts)
		.filter(pair => coauthorshipCounts[pair] >= 2)
		.map(pair => {
			const [source, target] = pair.split(',');
			return {
				source,
				target,
				value: coauthorshipCounts[pair],
			};
		});

	const institutionCounts: Record<string, number> = {};
	const institutionCategories: Record<string, string> = {};
	const institutionCoCounts: Record<string, number> = {};
	const countryInstitutionCounts: Record<string, Record<string, number>> = {};

	Object.values(clusterMap).forEach(work => {
		const instNames = work.institutions.map(inst => inst.display_name);
		work.institutions.forEach(inst => {
			const instName = inst.display_name;
			const country = inst.country_code;
			institutionCounts[instName] = (institutionCounts[instName] || 0) + 1;
			if (!institutionCategories[instName]) {
				institutionCategories[instName] = country;
			}
			countryInstitutionCounts[country] = countryInstitutionCounts[country] || {};
			countryInstitutionCounts[country][instName] = (countryInstitutionCounts[country][instName] || 0) + 1;
		});

		for (let i = 0; i < instNames.length; i++) {
			for (let j = i + 1; j < instNames.length; j++) {
				const pair = [instNames[i], instNames[j]].sort().join(',');
				institutionCoCounts[pair] = (institutionCoCounts[pair] || 0) + 1;
			}
		}
	});

	const countryTotalCounts: Record<string, number> = {};
	Object.keys(countryInstitutionCounts).forEach(country => {
		countryTotalCounts[country] = Object.values(countryInstitutionCounts[country]).reduce((sum, count) => sum + count, 0);
	});
	const top6Countries = Object.keys(countryTotalCounts)
		.sort((a, b) => countryTotalCounts[b] - countryTotalCounts[a])
		.slice(0, 6);

	const selectedInstitutions = new Set<string>();
	top6Countries.forEach(country => {
		const insts = Object.entries(countryInstitutionCounts[country] || {})
			.filter(([_, count]) => count >= 2)
			.sort((a, b) => b[1] - a[1])
			.slice(0, 10)
			.map(([inst]) => inst);
		insts.forEach(inst => selectedInstitutions.add(inst));
	});

	const institutionNodes: InstitutionNode[] = Object.keys(institutionCounts)
		.filter(inst => institutionCounts[inst] >= 2 && selectedInstitutions.has(inst))
		.map(inst => ({
			name: inst,
			category: institutionCategories[inst],
		}));

	const institutionLinks: InstitutionLink[] = Object.keys(institutionCoCounts)
		.filter(pair => institutionCoCounts[pair] >= 2)
		.map(pair => {
			const [source, target] = pair.split(',');
			if (source !== target && selectedInstitutions.has(source) && selectedInstitutions.has(target)) {
				return {
					source,
					target,
					value: institutionCoCounts[pair],
				};
			}
			return null;
		})
		.filter((link): link is InstitutionLink => link !== null);

	const conceptCounts: Record<string, number> = {};
	const conceptOriginalNames: Record<string, string> = {};
	works.forEach(work => {
		if (work.concepts && Array.isArray(work.concepts)) {
			work.concepts.forEach(concept => {
				if (concept.display_name) {
					const normalized = concept.display_name.toLowerCase();
					conceptCounts[normalized] = (conceptCounts[normalized] || 0) + 1;
					if (!conceptOriginalNames[normalized]) {
						conceptOriginalNames[normalized] = concept.display_name;
					}
				}
			});
		}
	});

	const topConcepts: Concept[] = Object.entries(conceptCounts)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 20)
		.map(([normalized, count]) => ({
			name: conceptOriginalNames[normalized],
			value: count,
		}));

	const keywordCounts: Record<string, number> = {};
	const keywordOriginalNames: Record<string, string> = {};
	works.forEach(work => {
		if (work.keywords && Array.isArray(work.keywords)) {
			work.keywords.forEach(keywordObj => {
				if (keywordObj.display_name) {
					const normalized = keywordObj.display_name.toLowerCase();
					keywordCounts[normalized] = (keywordCounts[normalized] || 0) + 1;
					if (!keywordOriginalNames[normalized]) {
						keywordOriginalNames[normalized] = keywordObj.display_name;
					}
				}
			});
		}
	});

	const topKeywords: Keyword[] = Object.entries(keywordCounts)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 20)
		.map(([normalized, count]) => ({
			name: keywordOriginalNames[normalized],
			value: count,
		}));

	const topArticles: Article[] = works
		.filter(work => work.cited_by_count !== undefined && work.cited_by_count !== null)
		.sort((a, b) => {
			if (b.cited_by_count !== a.cited_by_count) {
				return b.cited_by_count - a.cited_by_count;
			}
			return a.id.localeCompare(b.id);
		})
		.slice(0, 6)
		.map(work => ({
			id: work.id,
			display_name: work.display_name || `Untitled Article ${work.id}`,
			cited_by_count: work.cited_by_count || 0,
			publication_year: work.publication_year || 2021,
			counts_by_year: work.counts_by_year || [],
		}));

	const headers = ['2021', '2022', '2023', '2024', '2025'];
	const citationsPerYear: CitationsPerYear = {
		headers,
		values: {},
	};

	topArticles.forEach(article => {
		const titleKey = article.display_name.replace(/\n/g, ' ');
		const yearlyCounts = new Array(headers.length).fill(0);

		if (article.counts_by_year.length > 0) {
			article.counts_by_year.forEach(count => {
				const yearIndex = headers.indexOf(count.year.toString());
				if (yearIndex !== -1 && count.year >= Math.max(article.publication_year, 2021)) {
					yearlyCounts[yearIndex] = count.cited_by_count || 0;
				}
			});
		} else {
			const startYear = Math.max(article.publication_year, 2021);
			const yearsActive = Math.max(1, 2025 - startYear + 1);
			const avgCitations = Math.floor(article.cited_by_count / yearsActive);
			for (let year = startYear; year <= 2025; year++) {
				const yearIndex = headers.indexOf(year.toString());
				if (yearIndex !== -1) {
					yearlyCounts[yearIndex] = avgCitations;
				}
			}
			const totalAssigned = yearlyCounts.reduce((sum, count) => sum + count, 0);
			if (totalAssigned < article.cited_by_count) {
				const diff = article.cited_by_count - totalAssigned;
				const lastYearIndex = headers.indexOf('2025');
				if (lastYearIndex !== -1) {
					yearlyCounts[lastYearIndex] += diff;
				}
			}
		}

		citationsPerYear.values[titleKey] = yearlyCounts;
	});

	return {
		calendar_publications: { calendar },
		'pie-chart_countries': { values: topCountryCounts },
		'hierarchical-edge-bundling_co-authorship': { nodes: authorNodes, links: authorLinks },
		'force-directed-graph_co-authorship': { nodes: authorNodes, links: authorLinks },
		'hierarchical-edge-bundling_institutions': { nodes: institutionNodes, links: institutionLinks },
		'force-directed-graph_institutions': { nodes: institutionNodes, links: institutionLinks },
		'word-cloud_concepts': { words: topConcepts },
		'word-cloud_keywords': { words: topKeywords },
		'bar-chart_citations': citationsPerYear,
	};
}

export class ArticleToIllustry implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Article to Illustry',
		name: 'articleToIllustry',
		group: ['transform'],
		icon: { light: 'file:Illustry.svg', dark: 'file:Illustry.dark.svg' },
		version: 1,
		description: 'Processes OpenAlex publication data from a previous node to generate visualizations and save to Illustry API',
		defaults: { name: 'ArticleToIllustry' },
		inputs: [NodeConnectionType.Main],
		outputs: [NodeConnectionType.Main],
		properties: [
			{
				displayName: 'Illustry API URL',
				name: 'illustryApiUrl',
				type: 'string',
				default: 'http://illustrybackend:7001',
				required: true,
				description: 'Base URL for the Illustry API (e.g., http://your-api-host)',
				placeholder: 'http://your-api-host',
			},
			{
				displayName: 'Project Name',
				name: 'projectName',
				type: 'string',
				default: 'Articles',
				required: true,
				description: 'Name of the project for metadata and API payload',
				placeholder: 'Sustainability Research',
			},
			{
				displayName: 'Dashboard Name',
				name: 'dashboardName',
				type: 'string',
				default: 'Dashboard for articles',
				required: true,
				description: 'Name of the Dashboard for metadata and API payload',
				placeholder: 'Sustainability Research',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				const illustryApiUrl = this.getNodeParameter('illustryApiUrl', i, '') as string;
				const projectName = this.getNodeParameter('projectName', i, '') as string;
				const dashboardName = this.getNodeParameter('dashboardName', i, '') as string;

				if (!projectName) {
					throw new ApplicationError('Project Name is required');
				}
				if (!illustryApiUrl) {
					throw new ApplicationError('Illustry API URL is required');
				}

				// Extract binary data from OpenAlex node
				const binaryData = items[i].binary?.data;
				if (!binaryData) {
					throw new ApplicationError('No binary data found in input');
				}

				// Convert Buffer to string and parse JSON
				const binaryBuffer = await this.helpers.getBinaryDataBuffer(i, 'data');
				const works: WorkResponse[] = JSON.parse(binaryBuffer.toString('utf8'));

				if (!Array.isArray(works) || works.length === 0) {
					throw new ApplicationError('No valid publication data found in binary input');
				}

				// Create project via API
				const projectRequestOptions: IHttpRequestOptions = {
					method: 'POST' as IHttpRequestMethods,
					url: `${illustryApiUrl}/api/project`,
					headers: { 'Content-Type': 'application/json' },
					body: {
						projectName,
						projectDescription: 'Articles from OpenAlex',
						isActive: true,
					},
					json: true,
				};
				const projectData = await this.helpers.httpRequest(projectRequestOptions);
				// Generate visualizations
				const topClusters = extractTopPrimaryTopicClusters(works);
				const result = await createVisualizations(works, topClusters);

				// Store visualizations and prepare dashboard visualizations object
				const visualizationResponses = [];
				const dashboardVisualizations: Record<string, string> = {};

				// Save each visualization to Illustry API and build dashboard mapping
				for (const visualizationKey of Object.keys(result)) {
					// Split visualizationKey (e.g., 'pie-chart_countries') into type and name
					const [type, name] = visualizationKey.split('_');
					if (!type || !name) {
						// Skip if the key format is unexpected
						continue;
					}

					// Construct visualization payload for API
					const constructedVisualization = {
						data: {
							name, // e.g., 'countries'
							type, // e.g., 'pie-chart'
							projectName,
							description: `Visualization of ${name} data`,
							tags: [],
							data: result[visualizationKey as keyof typeof result] as Record<string, unknown>,
						},
					};

					// Send visualization to Illustry API
					const visualizationRequestOptions: IHttpRequestOptions = {
						method: 'POST' as IHttpRequestMethods,
						url: `${illustryApiUrl}/api/external/visualization`,
						headers: { 'Content-Type': 'application/json' },
						body: constructedVisualization,
						json: true,
					};
					const visualizationData = await this.helpers.httpRequest(visualizationRequestOptions);
					visualizationResponses.push(visualizationData);

					// Add to dashboard visualizations in format 'name(type)=type'
					// e.g., 'countries(pie-chart)': 'pie-chart'
					dashboardVisualizations[`${name}(${type})`] = type;
				}
				const dashboard = {
					name: dashboardName,
					description: 'Dashboard for articles from OpenAlex',
					projectName,
					visualizations: dashboardVisualizations, // e.g., { 'countries(pie-chart)': 'pie-chart', ... }
				};

				// Send dashboard to Illustry API
				const dashboardRequestOptions: IHttpRequestOptions = {
					method: 'POST' as IHttpRequestMethods,
					url: `${illustryApiUrl}/api/dashboard`, // Assumed endpoint for dashboard creation
					headers: { 'Content-Type': 'application/json' },
					body: dashboard,
					json: true,
				};
				await this.helpers.httpRequest(dashboardRequestOptions);
				returnData.push({
					json: {
						projectName,
						projectResponse: projectData,
						visualizationResponses,
					},
				});
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({ json: { error: `Error: ${error.message}` } });
				} else {
					throw new ApplicationError(`Error processing data: ${error.message}`);
				}
			}
		}

		return [returnData];
	}
}