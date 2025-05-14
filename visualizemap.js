import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import jsCookie from "js-cookie";
import { Alert, Snackbar } from "@mui/material";
import React, { useEffect, useState } from "react";
import Head from "next/head";
import FolderSharpIcon from '@mui/icons-material/FolderSharp';
import ArticleSharpIcon from '@mui/icons-material/ArticleSharp';

import { BASE_URL_CLIENT, WEBSITE_SEARCH_ENDPOINT } from "../static/constants";

export default function VisualizeMap() {
    // Declaration
    const router = useRouter();
    const [source, setSource] = useState("");
    const [query, setQuery] = useState("");
    const [communityId, setCommunityId] = useState("");
    const [height, setHeight] = useState(900);
    const [maxWidth, setMaxWidth] = useState(800);
    const [width, setWidth] = useState(800);
    const [path, setPath] = useState([]);

    const [tree, setTree] = useState([]);
    const [nodes, setNodes] = useState([]);
    const [visitedDocs, setVisitedDocs] = useState(new Set());
    const [visitedFolders, setVisitedFolders] = useState(new Set());

    // Necessary States for Alert Message
    const [open, setOpen] = useState(false);
    const [message, setMessage] = useState("");
    const [severity, setSeverity] = useState("error");

    //use Effect
    useEffect(() => {
        //  Set Canvas width and height
        let wd = window.innerWidth;
        let ht = window.innerHeight;
        setWidth(wd);
        setHeight(ht);
    }, []);

    useEffect(() => {
        if (!router.isReady) return;
        // Router is ready, safe to use router.query or other router properties/methods
        setQueryParams();
    }, [router.isReady, router.asPath]);

    // refresh the nodes at a level once the path changes
    useEffect(() => {
        if (tree.length) {
            setNodes(getNodes());
        }
    }, [tree, path]);

    // get the nodes that comprise a level
    const getNodes = () => {
        let level = tree;
        for (let idx of path) {
            level = level[idx].children;
        }
        return level;
    };

    const setQueryParams = () => {
        let obj = router.query;
        let src = "";
        let q = "";
        let cid = "";
        let ownSub = "False";

        if (obj != undefined || obj != null || obj != "") {
            src = obj["source"];
            q = obj["query"];
            cid = obj["community"];
            ownSub = obj["own_submissions"];

            if (q == undefined || q == null) q = "";
            if (cid == undefined || cid == null) cid = "all";
            if (ownSub == undefined || cid == null) ownSub = "";
        }
        setSource(src);
        setQuery(q);
        setCommunityId(cid);
        ownSub = ownSub.trim();
        getTreeData(q, cid, ownSub);
    };

    const getTreeData = async (query, communityId, ownSub) => {
        let url =
            BASE_URL_CLIENT +
            WEBSITE_SEARCH_ENDPOINT +
            "?query=" +
            // added encode - any queries with hashtags would fail on visualize
            encodeURIComponent(query) +
            "&community=" +
            communityId +
            "&source=website_visualize";
        if (ownSub == "True") {
            url += "&own_submissions=True";
        } else {
            url += "&own_submissions=False";
        }
        const res = await fetch(url, {
            method: "GET",
            headers: new Headers({
                Authorization: jsCookie.get("token"),
                "Content-Type": "application/json"
            })
        });
        const response = await res.json();
        console.log(response);
        if (response.status === "ok") {
            if ("tree" in response && response.tree) {
                setTree(response.tree);
            } else if ("error" in response && response.error) {
                setSeverity("error");
                setMessage(response.error);
                handleClick();
            } else {
                setSeverity("error");
                setMessage("Unexpected response format");
                handleClick();
            }
        } else {
            setSeverity("error");
            setMessage(response.message);
            handleClick();
        }
    };

    // For alerts
    const handleClick = () => {
        setOpen(true);
    };
    const handleClose = (event, reason) => {
        if (reason === "clickaway") {
            return;
        }
        setOpen(false);
    };

    const onClickItem = (node, idx) => {
        const currentPath = [...path, idx].join('/');
        if (node.type === 'node') {
            // set path as visited
            setVisitedFolders(new Set(visitedFolders).add(currentPath));
            // dive into this folder
            setPath([...path, idx]);
        } else if (node.type === 'leaf') {
            // set submission as visited
            const docId = node.docs[0];
            setVisitedDocs(new Set(visitedDocs).add(docId));
            // open the submission
            window.open(`https://textdata.org/submissions/${node.docs[0]}`, '_blank');
        }
    };

    // retrieve breadcrumb trail content
    const getBreadcrumbs = () => {
        const breadcrumbs = [];
        let level = tree;

        for (let i = 0; i < path.length; i++) {
            const idx = path[i];
            const node = level[idx];
            if (!node) break;
            breadcrumbs.push({ name: node.name, path: path.slice(0, i + 1) });
            level = node.children;
        }

        return breadcrumbs;
    };

    const setTrailPath = (path) => {
        // change path which refreshes nodes
        setPath(path);
        // manually clear inline hover styles from all breadcrumb spans
        // found this to be an issue where hover background would remain after clicking on an item in the trail
        document.querySelectorAll('.breadcrumb-item').forEach(el => {
            el.style.backgroundColor = 'transparent';
        });
    };

    if (!tree.length || !nodes.length) {
        return <>
            <Head>
                <title>Visualize - TextData</title>
                <link rel="icon" href="/images/tree32.png" />
            </Head>
            <div
                style={{ padding: 16 }}>
                Loading...
            </div>
            <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
                <Alert
                    onClose={handleClose}
                    severity={severity}
                    sx={{ width: "100%" }}
                >
                    {message}
                </Alert>
            </Snackbar>
        </>;
    }
    const shouldRenderIcons =
        Array.isArray(nodes) &&
        nodes.length > 0 &&
        nodes.some(node =>
            (node.name && node.name.length > 0) &&
            (node.docs?.length > 0 || node.type === "node")
        );
    return (
        <>
            <Head>
                <title>Visualize - TextData</title>
                <link rel="icon" href="/images/tree32.png" />
            </Head>
            <div style={{ padding: 16 }}>
                <div
                    style={{
                        width: '100%',
                        background: '#f5f5f5',
                        padding: '8px 24px',
                        boxSizing: 'border-box',
                        marginBottom: '12px',
                        fontSize: '14px',
                        display: 'flex',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                    }} >
                    <span
                        className="breadcrumb-item"
                        onClick={() => path.length !== 0 && setTrailPath([])}
                        style={{
                            cursor: path.length === 0 ? 'default' : 'pointer',
                            color: path.length === 0 ? 'black' : "#1976d2",
                            fontWeight: path.length === 0 ? 'bold' : 'normal',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            transition: 'background 0.2s ease-in-out',
                            textDecoration: 'none',
                            ...(path.length === 0
                                ? {}
                                : {
                                    ':hover': {
                                        backgroundColor: '#e0e0e0',
                                    },
                                }),
                        }}
                        onMouseEnter={e => {
                            if (path.length !== 0) e.currentTarget.style.backgroundColor = '#e0e0e0';
                        }}
                        onMouseLeave={e => {
                            if (path.length !== 0) e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                    >
                        Home
                    </span>
                    <div
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            height: '40px',
                            backgroundColor: '#f5f5f5',
                            padding: '0px',
                            fontFamily: 'sans-serif',
                        }}
                    >
                        {getBreadcrumbs().map((crumb, i, arr) => {
                            const isLast = i === arr.length - 1;
                            return (
                                <span key={i} style={{ display: 'flex', alignItems: 'center' }}>
                                    <span style={{ margin: '0 8px' }}>{'>'}</span>
                                    <span
                                        className="breadcrumb-item"
                                        onClick={() => !isLast && setTrailPath(crumb.path)}
                                        style={{
                                            cursor: isLast ? 'default' : 'pointer',
                                            color: isLast ? 'black' : "#1976d2",
                                            fontWeight: isLast ? 'bold' : 'normal',
                                            padding: '4px 8px',
                                            borderRadius: '4px',
                                            transition: 'background 0.2s ease-in-out',
                                            textDecoration: 'none',
                                            ...(isLast
                                                ? {}
                                                : {
                                                    ':hover': {
                                                        backgroundColor: '#e0e0e0',
                                                    },
                                                }),
                                        }}
                                        onMouseEnter={e => {
                                            if (!isLast) e.currentTarget.style.backgroundColor = '#e0e0e0';
                                        }}
                                        onMouseLeave={e => {
                                            if (!isLast) e.currentTarget.style.backgroundColor = 'transparent';
                                        }}
                                    >
                                        {crumb.name} ({crumb.path.reduce((nodes, idx, j) => {
                                            const n = nodes[idx];
                                            return j === crumb.path.length - 1 ? n.docs?.length || 0 : n.children;
                                        }, tree)})
                                    </span>
                                </span>
                            );
                        })}
                    </div>
                </div>
                {shouldRenderIcons ? (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                        gap: '16px',
                        justifyContent: 'start',
                        padding: '0 24px',
                    }}
                    >
                        {nodes.map((node, idx) => {
                            const currentPath = [...path, idx].join('/');
                            const isVisitedFolder = visitedFolders.has(currentPath);
                            const isVisitedDoc = visitedDocs.has(node.docs?.[0]);

                            return (
                                <div
                                    key={idx}
                                    onClick={() => onClickItem(node, idx)}
                                    style={{
                                        cursor: 'pointer',
                                        textAlign: 'center',
                                        padding: '8px',
                                        borderRadius: '6px',
                                        width: '100%',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        transition: 'background 0.2s ease-in-out',

                                        color: node.type === 'node'
                                            ? isVisitedFolder ? 'gray' : "#1976d2"
                                            : isVisitedDoc ? 'gray' : "#1976d2"
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.background = '#f0f0f0'}
                                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                >
                                    {node.type === 'node'
                                        ? <FolderSharpIcon style={{ fontSize: 48 }} />
                                        : <ArticleSharpIcon style={{ fontSize: 48 }} />}
                                    <div style={{
                                        display: '-webkit-box',
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: 'vertical',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        maxWidth: '140px',
                                        lineHeight: '1.2em',
                                        height: '2.4em',
                                        textAlign: 'center',
                                        color: node.type === 'node'
                                            ? isVisitedFolder ? 'gray' : 'black'
                                            : isVisitedDoc ? 'gray' : 'black'
                                    }}
                                        title={node.name}>
                                        {node.type === 'node'
                                            ? (
                                                <>
                                                    {node.name} <br /> ({node.docs.length})
                                                </>
                                            )
                                            : node.name
                                        }
                                    </div>
                                </div>
                            );
                        })}
                    </div>) : (
                    <div style={{
                        padding: '1rem 24px',
                        fontStyle: 'italic',
                        color: '#999',
                    }}>
                        No documents to display
                    </div>
                )}
            </div>
        </>
    );
}