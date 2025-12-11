const { readFileSync, writeFileSync } = require('fs');
const axios = require('axios');
const { OpenAlex } = require('openalex-ts');
const { v4: uuidv4 } = require('uuid');

const openAlex = OpenAlex.getInstance({
	openalexUrl: 'https://api.openalex.org',
	email: 'test-email@outlook.com',
	userAgent: 'openalex-ts/1.0.0',
	maxRetries: 3,
	retryBackoffFactor: 0.2,
	retryHttpCodes: [429, 500, 503],
});

function parseTSV(fileContent) {
	const lines = fileContent.split('\n').filter(line => line.trim());
	const headers = lines[0].split('\t').map(header => header.trim());
	return lines.slice(1).map(line => {
		const values = line.split('\t').map(value => value.trim());
		const record = {};
		headers.forEach((header, index) => {
			record[header] = values[index] || '';
		});
		return record;
	});
}

function readBibData(filePath) {
	return parseTSV(readFileSync(filePath, 'utf-8'));
}

function createInvertedIndex(abstract) {
	if (!abstract) return {};
	const words = abstract.split(/\s+/).filter(Boolean);
	const index = {};
	words.forEach((word, i) => {
		word = word.replace(/[^\w\s-]/g, '').toLowerCase();
		if (word) {
			index[word] = index[word] || [];
			index[word].push(i);
		}
	});
	return index;
}

function getAuthorLastname(name, isTSV = false) {
	if (!name) return '';
	if (isTSV) {
		const parts = name.split(',').map(part => part.trim()).filter(Boolean);
		return parts[0].toLowerCase();
	}
	const words = name.split(/\s+/).filter(Boolean);
	return words.length ? words[words.length - 1].toLowerCase() : '';
}

async function sendHttpPost(article, journal, abstract) {
	const payload = [{
		title: article.title,
		abstract_inverted_index: createInvertedIndex(abstract),
		inverted: true,
		referenced_works: [],
		journal_display_name: journal || 'Unknown Journal'
	}];
	try {
		const response = await axios.post('http://localhost:8080/invocations', payload, {
			headers: { 'Content-Type': 'application/json' },
			timeout: 60000
		});
		return response.data;
	} catch (error) {
		return { error: error.message };
	}
}

async function searchOpenAlex(title, authors) {
	try {
		const filters = { 'title_and_abstract.search': title };
		const response = await openAlex
			.getWork()
			.filter(filters)
			.sort('relevance_score', 'desc')
			.get();

		const tsvAuthorLastnames = authors
			.split(';')
			.map(author => getAuthorLastname(author.trim(), true))
			.filter(Boolean);

		for (const work of response) {
			if (work.title && work.title.toLowerCase() === title.toLowerCase()) {
				const openAlexAuthorLastnames = (work.authorships || [])
					.map(authorship => getAuthorLastname(authorship.author?.display_name || '', false))
					.filter(Boolean);

				if (tsvAuthorLastnames.some(tsvLastname =>
					openAlexAuthorLastnames.some(openAlexLastname => openAlexLastname.includes(tsvLastname))
				)) {
					return { found: true, work };
				}
			}
		}
		return { found: false };
	} catch (error) {
		return { found: false };
	}
}

function constructWorkResponse(record, postResponse) {
	const id = `https://openalex.org/W${uuidv4().replace(/-/g, '')}`;
	const authors = record.AU ? record.AU.split(';').map(a => a.trim()).filter(Boolean) : ['Unknown Author'];
	const publicationYear = parseInt(record.PY, 10) || 2021;
	const topics = Array.isArray(postResponse) && postResponse[0] ? postResponse[0] : [];
	const primaryTopic = topics[0] || null;
	console.log('aici???????')
	return {
		id,
		display_name: record.TI,
		title: record.TI,
		publication_year: publicationYear,
		cited_by_count: 0,
		counts_by_year: [],
		// created_date: `${publicationYear}-01-01`,
		authorships: authors.map(author => ({
			author: { display_name: author },
			// institutions: [{ display_name: 'Unknown Institution', country_code: 'XX' }],
			// countries: ['XX']
		})),
		primary_topic: primaryTopic ? {
			id: `https://openalex.org/T${primaryTopic.topic_id}`,
			display_name: primaryTopic.topic_label.split(':')[1].trim(),
		} : null,
		concepts: topics.map(topic => ({
			id: `https://openalex.org/T${topic.topic_id}`,
			display_name: topic.topic_label.split(':')[1].trim(),
			score: topic.topic_score
		})),
		keywords: record.AB ? record.AB.split(/\s+/).filter(Boolean).slice(0, 10).map(word => ({
			display_name: word.replace(/[^\w\s-]/g, ''),
			score: 1.0
		})) : [],
		source: {
			display_name: record.SO
		}
	};
}

async function main(filePath) {
	const records = readBibData(filePath);
	const workResponses = [];

	for (const record of records) {
		const title = record.TI;
		const authors = record.AU || '';
		const journal = record.SO || '';
		const abstract = record.AB || '';

		if (!title || !authors) {
			const article = { title: title || '(Missing Title)', authors: authors || '(Missing Authors)' };
			const postResponse = await sendHttpPost(article, journal, abstract);
			workResponses.push(constructWorkResponse({ ...record, TI: title || '(Missing Title)', AU: authors || '(Missing Authors)' }, postResponse));
			continue;
		}

		const result = await searchOpenAlex(title, authors);
		if (result.found && result.work) {
			workResponses.push(result.work);
		} else {
			const article = { title, authors };
			const postResponse = await sendHttpPost(article, journal, abstract);
			workResponses.push(constructWorkResponse(record, postResponse));
		}
	}

	writeFileSync('openalex_articles.json', JSON.stringify(workResponses, null, 2));
}

const filePath = '/Users/vladimirnitu/workspace/JSOpenAlex/Articles.txt';
main(filePath);