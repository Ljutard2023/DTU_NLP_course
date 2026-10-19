# DTU_NLP_course
GitHub repo for the NLP course
# Persons_to_wikidata
The architectural design of the microservice introduces specific execution sequences and algorithmic choices to address the assignment's constraints. 

Entity linking is structurally decoupled from property extraction : the **wbsearchentities API** handles fuzzy name resolution first, passing the isolated QID to the SPARQL endpoint to optimize the query execution sequence. At the internal mechanics level, the service uses **synchronous HTTP calls** via requests inside standard def endpoints. FastAPI automatically delegates these to an external worker thread pool, preventing the main event loop from blocking during network latency. 

For the algorithmic parsing of complex properties like dates, the logic applies a **direct string slice ([:10])** to the primary SPARQL binding, intentionally abstracting away Julian/Gregorian calendar precision issues to strictly satisfy the requested 'YYYY-MM-DD' format. 

Finally, a unified **property-fetching module** was designed to integrate both optional endpoints while maintaining a strict Don't Repeat Yourself architecture.
