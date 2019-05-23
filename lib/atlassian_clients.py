from atlassian import Confluence, Jira
from .data_management import DataEmitter, KeyedDataEmitter, CachedDataProvider
from .text_processing import text_simplifier

class JiraSimplifiedTextProvider:
    def __init__(self, jira):
        self.remote = jira
        self._simplify_text = text_simplifier()

    @staticmethod
    def extract_full_text(raw_ticket):
        try:
            comments = raw_ticket['fields']['comment']['comments']
            comments = ' '.join([c['body'] for c in comments])
        except KeyError:
            comments = ''

        title = raw_ticket['fields']['summary']
        body = raw_ticket['fields']['description']
        if body is None:
            body = ''

        return title + ' ' + body + ' ' + comments

    def extract_simple_text(self, raw_ticket):
        return self._simplify_text(
            JiraSimplifiedTextProvider.extract_full_text(raw_ticket))

    def __getitem__(self, key):
        return self.extract_simple_text(self.remote[key])


class JiraManager:

    def __init__(self, credentials, path):
        self.remote = Jira(
            url=credentials[0],
            username=credentials[1],
            password=credentials[2])
        self.cached = CachedDataProvider(self, path)
        self.simplified = CachedDataProvider(
            JiraSimplifiedTextProvider(self.cached),
            path + '.simplified')

    @staticmethod
    def extract_project_name(raw_ticket):
        return raw_ticket['fields']['project']['key']

    @staticmethod
    def extract_ticket_type(raw_ticket):
        return raw_ticket['fields']['issuetype']['name']

    @staticmethod
    def extract_linked_jira_ids(raw):
        ret = []
        for l in raw['fields']['issuelinks']:
            if 'outwardIssue' in l:
                ret.append(l['outwardIssue']['key'])
            elif 'inwardIssue' in l:
                ret.append(l['inwardIssue']['key'])

        return ret

    def get_simple_text(self, id_):
        return self.simplified[id_]

    def get_tickets_ids_by_filter(self, jql):
        start = 0
        window = 200
        ret = []
        total = self.remote.jql(jql, limit=1, start=0, fields='key')['total']
        for start in range(0, total, window):
            keys_resp = self.remote.jql(jql, limit=window, start=start, fields='key')
            ret += [k['key'] for k in keys_resp['issues']]
        return ret

    def link_issue_to_issue(self, source_id, target_id, link_type="Linked", force=False):
        if not force:
            source_raw = self.cached[source_id]
            source_linked_ids = JiraManager.extract_linked_jira_ids(source_raw)
            if target_id in source_linked_ids:
                return

        json = {
            "type": {
                "name": link_type
            },
            "inwardIssue": {
                "key": source_id
            },
            "outwardIssue": {
                "key": target_id
            }
        }
        self.remote.post('/rest/api/latest/issueLink', json)

    def multi_link_issue_to_issue(self, id_pairs, link_type="Linked", refetch=True, force=False):
        if refetch:
            unique_source_ids = list(set([l[0] for l in id_pairs]))
            for id_ in unique_source_ids:
                self.cached.force_fetch(id_)

        for pair in id_pairs:
            self.link_issue_to_issue(*pair, link_type, force)

    def get_remote_links(self, id_):
        raw = self.remote.get('/rest/api/latest/issue/' + id_ + '/remotelink')
        ret = []
        for l in raw:
            if 'object' in l:
                if 'url' in l['object']:
                    ret.append(l['object']['url'])
        return ret

    def link_issue_to_remote(self,
                             source_id, target_url,
                             target_title,
                             target_id=None,
                             relationship=None,
                             target_icon=None,
                             force=False):
        if not force:
            existing_links = self.get_remote_links(source_id)
            if target_url in existing_links:
                return

        if target_id is None:
            target_id = target_url

        def construct_remote_link_data():
            obj = {
                'url': target_url,
                'title': target_title,
            }
            if target_icon is not None:
                obj['icon'] = {
                    'url16x16': target_icon,
                    'title': 'Confluence'
                }

            data = {"object": obj}
            data['globalId'] = 'lnk_' + source_id + '_' + target_id
            if relationship is not None:
                data['relationship'] = relationship

            return data

        data = construct_remote_link_data()

        self.remote.post('/rest/api/latest/issue/' + source_id + '/remotelink', data)
        
    def issue_history(self, id_):
        def issue(self, key, fields='*all', expand=''):
                return self.get('rest/api/2/issue/{0}?fields={1}&expand={2}'\
                                .format(key, fields, expand))

        def extract_fields(h):
            a = h['author']['name']
            c = h['created']
            return (c, a, [e['field'] for e in h['items']])

        raw = issue(self.remote, id_, expand='changelog')

        return [extract_fields(h) for h in raw['changelog']['histories']]

    def __getitem__(self, key):
        return self.remote.issue(key)


class ConfluenceSimplifiedTextProvider:

    def __init__(self, confluence):
        self.remote = confluence
        self._simplify_text = text_simplifier()

    @staticmethod
    def extract_full_text(raw_page):
        title = raw_page['title']
        try:
            body = raw_page['body']['storage']['value']
        except KeyError:
            body = ''

        return title + ' ' + body

    def extract_simple_text(self, raw_page):
        return self._simplify_text(
            ConfluenceSimplifiedTextProvider.
                extract_full_text(raw_page))

    def __getitem__(self, key):
        return self.extract_simple_text(self.remote[key])


class ConfluenceManager:

    def __init__(self, credentials, path):
        self.remote = Confluence(
            url=credentials[0],
            username=credentials[1],
            password=credentials[2])

        def converter_to_local_key(key):
            if isinstance(key, tuple):
                return 'name_' + key[0] + key[1]
            else:
                return 'id_' + key

        self.cached = CachedDataProvider(self,
                                         path,
                                         converter_to_local_key=converter_to_local_key)
        self.simplified = CachedDataProvider(
            ConfluenceSimplifiedTextProvider(self.cached),
            path + '.simplified',
            converter_to_local_key=converter_to_local_key)

    def extract_url(self, raw_page):
        return self.remote.url + raw_page['_links']['webui']

    def extract_title(self, raw_page):
        return raw_page['title']

    def __get_page_id(self, id_):
        if isinstance(id_, tuple):
            id_ = self.remote.get_page_id(id_[0], id_[1])
        return id_

    def get_page(self, id_, full_info=True):
        id_ = self.__get_page_id(id_)

        expand = 'children.page.id'
        if full_info:
            expand = 'version,body.storage,children.page.id'

        raw = self.remote.get_page_by_id(id_, expand=expand)
        return raw

    def get_simple_text(self, id_):
        return self.simplified[id_]

    def get_page_tree_ids(self, id_):
        page = self.get_page(id_, full_info=False)
        ret = [page['id']]
        children_ids = [r['id'] for r in page['children']['page']['results']]
        for id_ in children_ids:
            ret += self.get_page_tree_ids(id_)
        return ret

    def __getitem__(self, key):
        if isinstance(key, tuple):
            remote_key = self.remote.get_page_id(key[0], key[1])
        else:
            remote_key = key

        ret = self.remote.get_page_by_id(remote_key, expand='version,body.storage,children.page.id')
        return ret


class JiraKeysEmitter(DataEmitter):

    def __init__(self, jira, jql):
        self.jira = jira
        self.jql = jql

    def emit(self):
        print('Emitting Jira keys')
        return self.jira.get_tickets_ids_by_filter(self.jql)


class ConfluenceKeysEmitter(DataEmitter):

    def __init__(self, confluence, root_pages_ids):
        self.confluence = confluence
        self.root_pages_ids = root_pages_ids

    def emit(self):
        keys = []
        print('Emitting Confluence keys')
        for page_id in self.root_pages_ids:
            keys += self.confluence.get_page_tree_ids(page_id)
        return list(set(keys))

def ConfluenceCorpusEmitter(confluence, root_pages_ids):
    pages_keys_emitter = ConfluenceKeysEmitter(confluence, root_pages_ids)
    corpus_emitter = KeyedDataEmitter(confluence.simplified, pages_keys_emitter)
    return corpus_emitter

def JiraCorpusEmitter(jira, jql):
    issues_keys_emitter = JiraKeysEmitter(jira, jql)
    corpus_emitter = KeyedDataEmitter(jira.simplified, issues_keys_emitter)
    return corpus_emitter

def link_jira_issue_to_confluence(jira,
                            confluence,
                            jira_id, confl_id, relationship=None):

    page_raw = confluence.cached[confl_id]
    page_title = confluence.extract_title(page_raw)
    page_url = confluence.extract_url(page_raw)

    jira.link_issue_to_remote(jira_id,
                                   page_url,
                                   page_title,
                                   confl_id,
                                   relationship,
                                   confluence.remote.url + '/images/icons/favicon.png')

def multy_link_jira_issue_to_confluence(jira, confluence, id_pairs, relationship, refetch=True):
    if refetch:
        unique_source_ids = list(set([l[0] for l in id_pairs]))
        for id_ in unique_source_ids:
            jira.cached.force_fetch(id_)

    for pair in id_pairs:
        link_jira_issue_to_confluence(jira, confluence, *pair, relationship)