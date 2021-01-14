"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for extracting the lead from pdfs

 Copyright 2020 The Authors. All Rights Reserved.
"""
import logging
import requests
import os


class RestClientForCronicle:
    """
        Rest Client for requests to Marburg Cronicle DB
        :param username: cronicle user
        :param password: password
        :param url: server URL
        :param port: server port
    """
    def __init__(self,
                 username,
                 password,
                 url='http://localhost:8080/native',
                 port=8080):

        self.username = username
        self.passwort = password
        self.url = url
        self.port = port

    def set_username(self, new_username):
        self.username = new_username

    def set_password(self, new_password):
        self.passwort = new_password

    def set_url(self, new_url):
        self.url = new_url

    def set_port(self, new_port):
        self.port = new_port

    def get_streams(self):
        uri = str(self.url) + '/get-streams'
        res = requests.post(uri, verify=False)

        return res.ok, res.json()

    def get_stream_info(self, stream_name):
        uri = str(self.url) + '/stream-info'
        data = {'name': stream_name}
        res = requests.post(uri, json=data, verify=False)

        return res.ok, res.json()

    def create_stream(self, stream_name, attributes, datatype='DOUBLE'):
        uri = str(self.url) + '/create-stream'
        schema = [{'name': 'TIMESTEP', 'type': 'LONG', 'properties': {'nullable': 'false'}}]

        for attr in attributes:
            schema.append({'name': attr, 'type': datatype, 'properties': {'nullable': 'false'}})

        print(schema)

        data = {'streamName': stream_name, 'schema': schema}
        res = requests.post(uri, json=data, verify=False)

        return res.ok

    def insert_events(self, stream_name, event_table):
        uri = str(self.url) + '/insert'
        events = []

        event_count = 5000

        for t in range(event_count):
            event = {}

            for column in event_table:
                event[column] = float(event_table[column][t])

            event['TSTART'] = float(t)
            event['TIMESTEP'] = float(t)
            events.append(event)

        data = {'streamName': stream_name, 'events': events}
        res = requests.post(uri, json=data, verify=False)
        if res.ok is not True:
            raise Exception('Events to t={} could not be inserted. {}'.format(event_count, res.json()))

    def make_query(self, query_string):
        uri = str(self.url) + '/query'
        query = {'queryString': query_string}

        res = requests.post(uri, json=query, verify=False)

        return res.json()



